"""Smart Spending Insights: deterministic analysis of a transaction DataFrame.

Everything here is plain Pandas/NumPy. All figures, percentages and flags are computed
from the user's actual data. Nothing in this module calls an LLM.

Methods (kept simple on purpose):
  * Month-over-month comparison per category, with a minimum size and percentage change.
  * Outliers: Tukey IQR fences, per category when there is enough history, otherwise against all
    expenses with stricter limits. An outlier is NOT a claim of fraud.
  * Recurring payments: same normalised merchant, >= 3 occurrences, regular intervals
    (weekly / biweekly / monthly / quarterly / yearly) and a stable amount.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from src import budget_analyzer as ba
from src.financial_calculations import format_inr as inr

REQUIRED_COLUMNS = ["date", "description", "category", "type", "amount"]
REFUND_RE = re.compile(r"refund|reversal|chargeback|cashback|money back|reimburs", re.IGNORECASE)
UNATTRIBUTED = "(Unattributed refunds)"
UNCATEGORIZED = "Uncategorized"

SEVERITY_ORDER = {"high": 0, "medium": 1, "positive": 2, "info": 3}
CADENCES = [  # (name, min_days, max_days, payments_per_year)
    ("weekly", 6, 8, 52), ("every 2 weeks", 13, 16, 26), ("monthly", 26, 35, 12),
    ("quarterly", 84, 98, 4), ("yearly", 350, 380, 1),
]


@dataclass(frozen=True)
class InsightConfig:
    change_pct_threshold: float = 20.0       # category change needed to be reported
    min_change_amount: float = 500.0         # ...and it must also be at least this many rupees
    overall_change_pct_threshold: float = 10.0
    outlier_iqr_k: float = 1.5               # Tukey fence when the category has enough history
    outlier_min_ratio: float = 2.0           # ...and amount must be >= this x the category median
    outlier_overall_iqr_k: float = 3.0       # stricter fence when comparing with all expenses
    outlier_overall_min_ratio: float = 3.0
    outlier_min_amount: float = 500.0
    min_category_history: int = 8
    min_overall_history: int = 15
    max_unusual: int = 10
    recurring_min_occurrences: int = 3
    recurring_regular_share: float = 0.70    # share of intervals that must be close to the median interval
    amount_cv_fixed: float = 0.05            # coefficient of variation: <= this means "fixed amount"
    variable_cv_max: float = 0.25            # varying-amount patterns (e.g. utility bills) must stay within this
    variable_min_occurrences: int = 4        # ...and need more evidence than fixed payments
    variable_regular_share: float = 0.80
    price_change_pct: float = 10.0
    partial_month_ratio: float = 0.85


@dataclass
class Insight:
    kind: str                    # trend_increase | trend_decrease | new_category | overspend | overall | price_change
    severity: str                # high | medium | positive | info
    title: str
    message: str                 # deterministic explanation, written from the numbers below
    suggestion: str
    category: Optional[str] = None
    figures: list = field(default_factory=list)   # [(label, formatted value)] for display
    values: dict = field(default_factory=dict)    # raw numbers behind the message
    details: list = field(default_factory=list)   # e.g. largest transactions behind a change (local display only)


@dataclass
class InsightReport:
    overview: dict
    category_table: pd.DataFrame
    insights: list
    unusual: pd.DataFrame
    recurring: pd.DataFrame
    notes: list
    unusual_note: str = ""
    quality: dict = field(default_factory=dict)

    @property
    def has_data(self) -> bool:
        return self.overview.get("transaction_count", 0) > 0


# ---------------------------------------------------------------- cleaning
def merchant_key(description: str) -> str:
    """Normalise a description so 'NETFLIX 1234' and 'Netflix' group together."""
    text = re.sub(r"[^a-z\s]", " ", str(description).casefold())
    text = re.sub(r"\s+", " ", text).strip()
    return text or str(description).casefold().strip()


def _normalize_categories(series: pd.Series) -> tuple[pd.Series, dict]:
    cleaned = series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    cleaned = cleaned.where(cleaned != "", UNCATEGORIZED)
    key = cleaned.str.casefold()
    def pick(group: pd.Series) -> str:
        counts = group.value_counts()
        return sorted(counts.index, key=lambda label: (-counts[label], label != label.title(), label))[0]

    canonical = {k: pick(grp) for k, grp in cleaned.groupby(key)}
    mapped = key.map(canonical).astype(str)
    merged = {orig: canonical[orig.casefold()] for orig in cleaned.unique() if orig != canonical[orig.casefold()]}
    return mapped, merged


def prepare_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean a transaction table and build the 'spending' frame.

    Returns (spend, quality). ``spend`` has one row per expense (signed amount > 0) and per refund
    (signed amount < 0); other income is excluded so it never counts as spending.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))

    work = df[REQUIRED_COLUMNS].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    work["type"] = work["type"].astype(str).str.strip().str.lower()
    work["description"] = work["description"].fillna("").astype(str).str.strip()
    bad = work["date"].isna() | work["amount"].isna() | ~np.isfinite(work["amount"].astype(float)) \
        | (work["amount"] <= 0) | ~work["type"].isin(["income", "expense"])
    quality = {"records": int(len(work)), "invalid_dropped": int(bad.sum())}
    work = work[~bad].copy()
    work["date"] = work["date"].dt.normalize()
    work["category"], merged = _normalize_categories(work["category"])
    work["category"] = work["category"].astype(str)
    quality["merged_category_labels"] = merged
    quality["possible_duplicates"] = int(work.duplicated(subset=REQUIRED_COLUMNS, keep="first").sum())

    expenses = work[work["type"] == "expense"].copy()
    expenses["kind"], expenses["signed"] = "expense", expenses["amount"]

    is_refund = (work["type"] == "income") & (
        work["category"].str.contains(REFUND_RE) | work["description"].str.contains(REFUND_RE))
    refunds = work[is_refund].copy()
    known = {c.casefold(): c for c in expenses["category"].unique()}

    def resolve(row) -> str:
        if row["category"].casefold() in known:
            return known[row["category"].casefold()]
        for folded, original in known.items():          # e.g. "Refund - Shopping order"
            if folded in row["description"].casefold():
                return original
        return UNATTRIBUTED

    if not refunds.empty:
        refunds["category"] = refunds.apply(resolve, axis=1)
    refunds["kind"], refunds["signed"] = "refund", -refunds["amount"]
    quality["refunds"] = int(len(refunds))
    quality["income_excluded"] = int(((work["type"] == "income") & ~is_refund).sum())

    spend = pd.concat([expenses, refunds], ignore_index=True)
    spend["month"] = spend["date"].dt.strftime("%Y-%m")
    spend["merchant"] = spend["description"].map(merchant_key)
    return spend.sort_values("date", kind="stable").reset_index(drop=True), quality


# ---------------------------------------------------------------- comparisons
def _prev_month(month: str) -> str:
    return str(pd.Period(month, freq="M") - 1)


def category_comparison(spend: pd.DataFrame, latest: str, previous: str) -> pd.DataFrame:
    pivot = spend.pivot_table(index="category", columns="month", values="signed", aggfunc="sum").fillna(0.0)
    for m in (latest, previous):
        if m not in pivot.columns:
            pivot[m] = 0.0
    out = pd.DataFrame({"category": pivot.index, "previous": pivot[previous].values, "latest": pivot[latest].values})
    out["change"] = out["latest"] - out["previous"]
    out["change_pct"] = np.where(out["previous"] > 0, out["change"] / out["previous"].where(out["previous"] > 0) * 100, np.nan)

    def status(r) -> str:
        if r.previous <= 0 and r.latest > 0:
            return "new"
        if r.latest <= 0 < r.previous:
            return "stopped"
        return "up" if r.change > 0 else "down" if r.change < 0 else "flat"

    out["status"] = [status(r) for r in out.itertuples()]
    out = out.sort_values(["latest", "previous"], ascending=False).reset_index(drop=True)
    return out


def _top_transactions(spend: pd.DataFrame, category: str, month: str, n: int = 3) -> list:
    g = spend[(spend["category"] == category) & (spend["month"] == month) & (spend["kind"] == "expense")]
    return [f"{r.description} ({inr(r.amount)} on {r.date:%d %b})" for r in g.nlargest(n, "amount").itertuples()]


def _trend_insights(spend, table, latest, previous, partial, budgets, cfg) -> list:
    insights: list[Insight] = []
    months_before = sorted(m for m in spend["month"].unique() if m < latest)[-3:]
    pivot = spend.pivot_table(index="category", columns="month", values="signed", aggfunc="sum").fillna(0.0)

    for r in table.itertuples():
        if r.category == UNATTRIBUTED:
            continue
        baseline = float(pivot.loc[r.category, months_before].mean()) if months_before else 0.0
        figs = [(previous, inr(r.previous)), (latest, inr(r.latest)), ("Change", f"{'+' if r.change >= 0 else '-'}{inr(abs(r.change))}")]
        vals = {"previous": r.previous, "latest": r.latest, "change": r.change, "baseline": baseline}
        if r.previous > 0:
            pct = r.change / r.previous * 100
            figs.append(("Change %", f"{pct:+.0f}%"))
            vals["change_pct"] = pct
            if pct >= cfg.change_pct_threshold and r.change >= cfg.min_change_amount:
                tail = (f" Your average over the {len(months_before)} month(s) before was {inr(baseline)}."
                        if len(months_before) >= 2 else "")
                insights.append(Insight(
                    "trend_increase", "high" if pct >= 50 else "medium", f"{r.category} spending is up {pct:.0f}%",
                    f"{r.category} spending increased by {pct:.0f}% compared with {previous} "
                    f"({inr(r.previous)} to {inr(r.latest)}, +{inr(r.change)}). Review your recent {r.category} "
                    f"transactions to understand the increase.{tail}",
                    f"Check the largest {r.category} transactions below and decide whether to set a monthly limit.",
                    r.category, figs, vals, _top_transactions(spend, r.category, latest)))
            elif pct <= -cfg.change_pct_threshold and -r.change >= cfg.min_change_amount and not partial:
                insights.append(Insight(
                    "trend_decrease", "positive", f"{r.category} spending is down {abs(pct):.0f}%",
                    f"{r.category} spending decreased by {abs(pct):.0f}% compared with {previous} "
                    f"({inr(r.previous)} to {inr(r.latest)}, -{inr(abs(r.change))}).",
                    "If this reduction is intentional, consider moving the saved amount into savings.",
                    r.category, figs, vals))
        elif r.latest >= cfg.min_change_amount:
            insights.append(Insight(
                "new_category", "info", f"New spending in {r.category}",
                f"You spent {inr(r.latest)} on {r.category} in {latest}, with no spending in this category in {previous}.",
                "Check whether this is a one-off or the start of a regular expense.",
                r.category, [(latest, inr(r.latest))], {"latest": r.latest}, _top_transactions(spend, r.category, latest)))

    if budgets:
        frame = spend.assign(type="expense", amount=spend["signed"])[["date", "category", "type", "amount"]]
        for r in ba.overspent_categories(ba.compare_budgets(frame, budgets, latest)).itertuples():
            pct_used = r.percent_used
            insights.append(Insight(
                "overspend", "high" if pct_used >= 125 else "medium", f"{r.category} is over budget",
                f"{r.category} spending in {latest} was {inr(r.actual)} against a monthly budget of {inr(r.budget)}: "
                f"over by {inr(-r.difference)} ({pct_used:.0f}% of the budget).",
                f"Reduce {r.category} by about {inr(-r.difference)} next month, or review whether the budget is realistic.",
                r.category, [("Budget", inr(r.budget)), ("Actual", inr(r.actual)), ("Used", f"{pct_used:.0f}%")],
                {"budget": r.budget, "actual": r.actual, "percent_used": pct_used}))
    return insights


# ---------------------------------------------------------------- recurring payments
def _cadence(median_days: float):
    for name, lo, hi, per_year in CADENCES:
        if lo <= median_days <= hi:
            return name, per_year
    return None


def detect_recurring(expenses: pd.DataFrame, as_of: pd.Timestamp, cfg: InsightConfig = InsightConfig()) -> pd.DataFrame:
    """Find payments that repeat at regular intervals with a stable amount (gross expenses only)."""
    columns = ["merchant", "category", "cadence", "pattern", "typical_amount", "occurrences", "est_monthly",
               "est_annual", "last_date", "next_expected", "usual_amount", "latest_amount", "status", "confidence", "indices"]
    rows = []
    for _, g in expenses.groupby("merchant"):
        g = g.sort_values("date").drop_duplicates(subset="date", keep="first")
        if len(g) < cfg.recurring_min_occurrences:
            continue
        intervals = np.diff(g["date"].values).astype("timedelta64[D]").astype(float)
        median_gap = float(np.median(intervals))
        cadence = _cadence(median_gap)
        if cadence is None:
            continue
        tolerance = max(3.0, 0.15 * median_gap)
        regular = float(np.mean(np.abs(intervals - median_gap) <= tolerance))
        if regular < cfg.recurring_regular_share:
            continue
        amounts = g["amount"].astype(float).values
        mean = float(amounts.mean())
        cv = float(amounts.std() / mean) if mean else 0.0
        fixed = cv <= cfg.amount_cv_fixed
        usual, latest_amount = float(np.median(amounts)), float(amounts[-1])
        if not fixed and len(amounts) >= 4:  # stable history followed by one changed payment = a price change
            core = amounts[:-1]
            core_mean = float(core.mean())
            core_cv = float(core.std() / core_mean) if core_mean else 0.0
            if core_cv <= cfg.amount_cv_fixed and abs(latest_amount - core_mean) / core_mean <= 0.5:
                fixed, usual = True, float(np.median(core))
        if not fixed and (cv > cfg.variable_cv_max or len(g) < cfg.variable_min_occurrences
                          or regular < cfg.variable_regular_share):
            continue
        changed = fixed and usual > 0 and abs(latest_amount - usual) / usual * 100 >= cfg.price_change_pct
        typical = (latest_amount if changed else usual) if fixed else mean
        name, per_year = cadence
        last = g["date"].max()
        rows.append({
            "merchant": g["description"].mode().iloc[0], "category": g["category"].mode().iloc[0],
            "cadence": name, "pattern": "Fixed amount" if fixed else "Variable amount",
            "typical_amount": typical, "occurrences": int(len(g)),
            "est_monthly": typical * per_year / 12, "est_annual": typical * per_year,
            "last_date": last, "next_expected": last + pd.Timedelta(days=round(median_gap)),
            "usual_amount": usual if fixed else mean, "latest_amount": latest_amount,
            "status": "Active" if (as_of - last).days <= 1.5 * median_gap else "May have ended",
            "confidence": "High" if len(g) >= 6 and regular >= 0.9 else "Medium",
            "indices": list(g.index),
        })
    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values("est_annual", ascending=False).reset_index(drop=True)


def _price_changes(recurring: pd.DataFrame, cfg: InsightConfig) -> list:
    out = []
    for r in recurring[(recurring["pattern"] == "Fixed amount") & (recurring["status"] == "Active")].itertuples():
        prior, last = float(r.usual_amount), float(r.latest_amount)
        pct = (last - prior) / prior * 100 if prior else 0.0
        if abs(pct) >= cfg.price_change_pct and abs(last - prior) >= 50:
            word = "increased" if pct > 0 else "decreased"
            out.append(Insight(
                "price_change", "medium" if pct > 0 else "positive", f"{r.merchant} payment {word}",
                f"The latest {r.merchant} payment was {inr(last)}, which is {abs(pct):.0f}% {'higher' if pct > 0 else 'lower'} "
                f"than its usual {inr(prior)}.", "Check whether the price change was expected.",
                r.category, [("Usual", inr(prior)), ("Latest", inr(last)), ("Change %", f"{pct:+.0f}%")],
                {"usual": prior, "latest": last, "change_pct": pct}))
    return out


# ---------------------------------------------------------------- unusual transactions
def detect_unusual(expenses: pd.DataFrame, exclude_index: Iterable = (), cfg: InsightConfig = InsightConfig()) -> tuple[pd.DataFrame, str]:
    """IQR outliers. Returns (table, note). Regular recurring payments are not tested."""
    columns = ["date", "description", "category", "amount", "typical", "upper_limit", "ratio", "scope", "reason"]
    ex = expenses.drop(index=list(exclude_index), errors="ignore")
    if len(ex) < cfg.min_overall_history and not (ex["category"].value_counts() >= cfg.min_category_history).any():
        return pd.DataFrame(columns=columns), (
            f"Not enough history to judge what is unusual (needs about {cfg.min_overall_history}+ expense transactions, "
            f"or {cfg.min_category_history}+ in one category). You have {len(ex)}.")

    def fence(values: pd.Series, k: float) -> tuple[float, float]:
        q1, q3 = np.percentile(values, [25, 75])
        return float(q3 + k * (q3 - q1)), float(np.median(values))

    overall_fence, overall_median = (fence(ex["amount"], cfg.outlier_overall_iqr_k) if len(ex) >= cfg.min_overall_history else (None, None))
    rows = []
    for cat, g in ex.groupby("category"):
        if len(g) >= cfg.min_category_history:
            upper, med = fence(g["amount"], cfg.outlier_iqr_k)
            ratio_min, scope, label = cfg.outlier_min_ratio, "category", f"{len(g)} {cat} transactions"
        elif overall_fence is not None:
            upper, med = overall_fence, overall_median
            ratio_min, scope, label = cfg.outlier_overall_min_ratio, "overall", f"all {len(ex)} expense transactions"
        else:
            continue
        hits = g[(g["amount"] > upper) & (g["amount"] >= ratio_min * med) & (g["amount"] >= cfg.outlier_min_amount)]
        for r in hits.itertuples():
            ratio = r.amount / med
            what = f"typical {cat} transaction" if scope == "category" else "typical expense"
            rows.append({
                "date": r.date, "description": r.description, "category": cat, "amount": float(r.amount),
                "typical": med, "upper_limit": upper, "ratio": ratio, "scope": scope,
                "reason": f"{inr(r.amount)} is {ratio:.1f}x the {what} (median {inr(med)}) and above the usual upper range "
                          f"of {inr(upper)}, based on {label}.",
            })
    out = pd.DataFrame(rows, columns=columns).sort_values("ratio", ascending=False).head(cfg.max_unusual)
    return out.reset_index(drop=True), "Statistical outliers only. They are not confirmed fraud or mistakes."


# ---------------------------------------------------------------- main entry point
def analyze(
    df: pd.DataFrame,
    budgets: Optional[Mapping[str, float]] = None,
    start=None,
    end=None,
    categories: Optional[Iterable[str]] = None,
    cfg: InsightConfig = InsightConfig(),
) -> InsightReport:
    """Run the full analysis on (optionally filtered) transactions."""
    spend, quality = prepare_transactions(df)
    notes: list[str] = []
    if quality["invalid_dropped"]:
        notes.append(f"{quality['invalid_dropped']} record(s) with an invalid date, amount or type were ignored.")
    if quality["merged_category_labels"]:
        pairs = ", ".join(f"'{a}' to '{b}'" for a, b in list(quality["merged_category_labels"].items())[:4])
        notes.append(f"Inconsistent category labels were merged ({pairs}).")
    if quality["possible_duplicates"]:
        notes.append(f"{quality['possible_duplicates']} transaction(s) look like exact duplicates (same date, description, "
                     "category and amount). They were kept because identical purchases can be genuine; check your file.")
    if quality["refunds"]:
        notes.append(f"{quality['refunds']} refund/credit transaction(s) were subtracted from spending instead of counted as income.")

    if start is not None:
        spend = spend[spend["date"] >= pd.Timestamp(start)]
    if end is not None:
        spend = spend[spend["date"] <= pd.Timestamp(end)]
    if categories is not None:
        spend = spend[spend["category"].isin(set(categories) | {UNATTRIBUTED})]
    spend = spend.reset_index(drop=True)

    empty_cols = ["category", "previous", "latest", "change", "change_pct", "status"]
    if spend.empty:
        notes.append("There are no expense transactions in the selected range.")
        return InsightReport({"transaction_count": 0}, pd.DataFrame(columns=empty_cols), [], detect_unusual(spend.iloc[0:0])[0],
                             detect_recurring(spend.iloc[0:0], pd.Timestamp.today()), notes, "", quality)

    months = sorted(spend["month"].unique())
    latest = months[-1]
    previous = _prev_month(latest)
    last_date = spend["date"].max()
    days_in_month = calendar.monthrange(int(latest[:4]), int(latest[5:]))[1]
    partial = last_date.day / days_in_month < cfg.partial_month_ratio
    comparison_available = previous in months

    gross = spend[spend["kind"] == "expense"]
    total_net = float(spend["signed"].sum())
    latest_total = float(spend.loc[spend["month"] == latest, "signed"].sum())
    prev_total = float(spend.loc[spend["month"] == previous, "signed"].sum()) if comparison_available else 0.0
    overview = {
        "transaction_count": int(len(gross)), "months": len(months), "total_spending": total_net,
        "gross_expenses": float(gross["amount"].sum()), "refunds_total": float(abs(spend.loc[spend["kind"] == "refund", "signed"].sum())),
        "avg_transaction": float(gross["amount"].mean()) if len(gross) else 0.0,
        "avg_monthly_spending": total_net / len(months), "latest_month": latest, "previous_month": previous,
        "latest_spending": latest_total, "previous_spending": prev_total, "comparison_available": comparison_available,
        "mom_change_amount": latest_total - prev_total if comparison_available else None,
        "mom_change_pct": (latest_total - prev_total) / prev_total * 100 if comparison_available and prev_total > 0 else None,
        "latest_month_partial": bool(partial), "start": spend["date"].min(), "end": last_date,
    }
    if partial:
        notes.append(f"Your data for {latest} ends on day {last_date.day} of {days_in_month}, so that month may be incomplete. "
                     "Decreases are not reported for it, and comparisons may be misleading.")
    if not comparison_available:
        notes.append(f"There are no transactions in {previous}, so month-over-month comparisons are not available. "
                     "At least two consecutive months of data are needed.")

    table = category_comparison(spend, latest, previous) if comparison_available else pd.DataFrame(columns=empty_cols)
    table = table[table["category"] != UNATTRIBUTED].reset_index(drop=True)  # such refunds only reduce the total
    insights: list[Insight] = []
    if comparison_available:
        insights = _trend_insights(spend, table, latest, previous, partial, budgets or {}, cfg)
        pct = overview["mom_change_pct"]
        if pct is not None and abs(pct) >= cfg.overall_change_pct_threshold and not (partial and pct < 0):
            up = pct > 0
            insights.append(Insight(
                "overall", "medium" if up else "positive", f"Total spending {'rose' if up else 'fell'} {abs(pct):.0f}%",
                f"Your total spending in {latest} was {inr(latest_total)}, {'up' if up else 'down'} {abs(pct):.0f}% from "
                f"{inr(prev_total)} in {previous}.", "See the category comparison to find which categories drive the change.",
                None, [(previous, inr(prev_total)), (latest, inr(latest_total)), ("Change %", f"{pct:+.0f}%")],
                {"previous": prev_total, "latest": latest_total, "change_pct": pct}))

    recurring = detect_recurring(gross, last_date, cfg)
    insights += _price_changes(recurring, cfg)
    unusual, unusual_note = detect_unusual(gross, exclude_index=[i for lst in recurring["indices"] for i in lst], cfg=cfg)
    insights.sort(key=lambda i: (SEVERITY_ORDER[i.severity], -abs(i.values.get("change", i.values.get("actual", 0.0)))))
    return InsightReport(overview, table, insights, unusual, recurring, notes, unusual_note, quality)


def active_fixed(recurring: pd.DataFrame) -> pd.DataFrame:
    """Active recurring payments with a stable amount (the 'commitments' that are safe to total)."""
    if recurring.empty:
        return recurring
    return recurring[(recurring["status"] == "Active") & (recurring["pattern"] == "Fixed amount")]


# ---------------------------------------------------------------- text for people and for the LLM
def fallback_summary(report: InsightReport) -> str:
    """Deterministic plain-language summary (used when Gemini is unavailable)."""
    if not report.has_data:
        return "There are no expense transactions in the selected range, so there is nothing to summarise yet."
    o = report.overview
    lines = []
    if o["comparison_available"] and o["mom_change_pct"] is not None:
        direction = "up" if o["mom_change_pct"] > 0 else "down" if o["mom_change_pct"] < 0 else "unchanged"
        lines.append(f"**Overview:** You spent {inr(o['latest_spending'])} in {o['latest_month']}, {direction} "
                     f"{abs(o['mom_change_pct']):.0f}% from {inr(o['previous_spending'])} in {o['previous_month']}.")
    else:
        lines.append(f"**Overview:** Your average monthly spending is {inr(o['avg_monthly_spending'])} over {o['months']} month(s). "
                     "A month-over-month comparison needs two consecutive months of data.")
    if report.insights:
        lines.append("**Key findings:**")
        lines += [f"- {i.message}" for i in report.insights[:8]]
    else:
        lines.append("**Key findings:** No category changed significantly, and nothing exceeded a budget in the latest month.")
    lines.append(f"**Unusual transactions:** {len(report.unusual)} statistical outlier(s) flagged for review." if len(report.unusual)
                 else "**Unusual transactions:** none flagged.")
    fixed = active_fixed(report.recurring)
    if len(fixed):
        lines.append(f"**Recurring payments:** {len(fixed)} active fixed-amount payment(s), about {inr(fixed['est_monthly'].sum())} per month "
                     f"({inr(fixed['est_annual'].sum())} per year). Verify that you still want each one.")
    return "\n\n".join(lines)


def facts_text(report: InsightReport, include_merchants: bool = False) -> str:
    """The verified facts that may be sent to Gemini. Merchant names are included only on request."""
    o = report.overview
    lines = ["VERIFIED FACTS (calculated by Python from the user's transactions)"]
    if not report.has_data:
        return "\n".join(lines + ["No expense transactions are available."])
    lines.append(f"Period: {o['start']:%Y-%m-%d} to {o['end']:%Y-%m-%d} ({o['months']} month(s)); average monthly spending {inr(o['avg_monthly_spending'])}")
    if o["comparison_available"]:
        pct = "n/a" if o["mom_change_pct"] is None else f"{o['mom_change_pct']:+.1f}%"
        lines.append(f"Spending in {o['latest_month']}: {inr(o['latest_spending'])}; in {o['previous_month']}: {inr(o['previous_spending'])}; change {pct}")
    lines.append("FINDINGS:" if report.insights else "FINDINGS: none (no significant changes).")
    lines += [f"{n}. [{i.severity}] {i.message}" + (f" Largest transactions: {'; '.join(i.details)}." if include_merchants and i.details else "")
              for n, i in enumerate(report.insights[:10], 1)]
    if len(report.unusual):
        lines.append("UNUSUAL TRANSACTIONS (statistical outliers, not confirmed fraud):")
        for r in report.unusual.itertuples():
            who = f"{r.description}, " if include_merchants else ""
            lines.append(f"- {who}{r.category}: {r.reason}")
    fixed = active_fixed(report.recurring)
    if len(fixed):
        lines.append(f"RECURRING FIXED-AMOUNT PAYMENTS: {len(fixed)} active; estimated {inr(fixed['est_monthly'].sum())} per month, {inr(fixed['est_annual'].sum())} per year")
        for r in fixed.head(8).itertuples():
            who = f"{r.merchant} ({r.category})" if include_merchants else r.category
            lines.append(f"- {who}: {r.cadence}, about {inr(r.typical_amount)} per payment, {r.occurrences} occurrences")
    return "\n".join(lines)