"""Streamlit section: Smart Spending Insights (rendered inside the Financial Dashboard page)."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from src import charts
from src import insights as ins
from src.financial_calculations import format_inr as inr
from src.gemini_client import is_configured
from src.insights_ai import explain_insights
from src.prompts import build_insights_prompt

PLOT_CONFIG = {"displayModeBar": False}
COLORS = {"high": "#F43F5E", "medium": "#F59E0B", "positive": "#14B8A6", "info": "#6366F1"}
LABELS = {"high": "Needs attention", "medium": "Worth a look", "positive": "Good news", "info": "FYI"}
VISIBLE_INSIGHTS = 6


def _card(insight: ins.Insight) -> str:
    color = COLORS[insight.severity]
    chips = "".join(
        f'<span style="display:inline-block;margin:0.15rem 0.35rem 0.15rem 0;padding:0.15rem 0.6rem;border-radius:999px;'
        f'background:#f1f0ff;color:#3730a3;font-size:0.8rem;font-weight:600;">{html.escape(label)}: {html.escape(value)}</span>'
        for label, value in insight.figures
    )
    details = ""
    if insight.details:
        items = "; ".join(html.escape(d) for d in insight.details)
        details = f'<div style="margin-top:0.35rem;font-size:0.85rem;color:#4b5563;"><b>Largest transactions:</b> {items}</div>'
    return (
        f'<div style="background:rgba(255,255,255,0.92);border-left:6px solid {color};border-radius:14px;'
        f'padding:0.9rem 1.1rem;margin:0.55rem 0;box-shadow:0 4px 14px rgba(99,102,241,0.10);color:#1e1b4b;">'
        f'<div style="font-size:0.72rem;font-weight:700;letter-spacing:0.5px;color:{color};text-transform:uppercase;">'
        f'{LABELS[insight.severity]}</div>'
        f'<div style="font-size:1.05rem;font-weight:700;margin:0.1rem 0 0.3rem 0;">{html.escape(insight.title)}</div>'
        f'<div style="font-size:0.93rem;line-height:1.45;">{html.escape(insight.message)}</div>'
        f'<div style="margin-top:0.4rem;">{chips}</div>{details}'
        f'<div style="margin-top:0.35rem;font-size:0.88rem;color:#4338ca;"><b>Suggestion:</b> {html.escape(insight.suggestion)}</div>'
        f'</div>'
    )


def _overview(report: ins.InsightReport) -> None:
    o = report.overview
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Spending in {o['latest_month']}", inr(o["latest_spending"]))
    if o["comparison_available"] and o["mom_change_pct"] is not None:
        c2.metric(f"Change vs {o['previous_month']}", f"{o['mom_change_pct']:+.1f}%")
        c2.caption(f"{'+' if o['mom_change_amount'] >= 0 else '-'}{inr(abs(o['mom_change_amount']))} compared with {inr(o['previous_spending'])}")
    else:
        c2.metric(f"Change vs {o['previous_month']}", "n/a")
        c2.caption("Needs spending in the previous calendar month")
    c3.metric("Average monthly spending", inr(o["avg_monthly_spending"]))
    c3.caption(f"{o['transaction_count']} expense transactions, average {inr(o['avg_transaction'])} each")
    if o["refunds_total"] > 0:
        st.caption(f"Spending is shown net of {inr(o['refunds_total'])} in refunds.")


def _category_section(report: ins.InsightReport) -> None:
    o, table = report.overview, report.category_table
    st.markdown("##### Category analysis")
    if table.empty:
        st.info("A category comparison needs spending in two consecutive months.")
        return
    st.plotly_chart(charts.category_comparison_chart(table, o["previous_month"], o["latest_month"]),
                    width="stretch", config=PLOT_CONFIG)
    shown = table.rename(columns={"category": "Category", "previous": f"{o['previous_month']} (₹)", "latest": f"{o['latest_month']} (₹)",
                                  "change": "Change (₹)", "change_pct": "Change (%)", "status": "Trend"})
    st.dataframe(shown, hide_index=True, width="stretch", column_config={
        f"{o['previous_month']} (₹)": st.column_config.NumberColumn(format="%.0f"),
        f"{o['latest_month']} (₹)": st.column_config.NumberColumn(format="%.0f"),
        "Change (₹)": st.column_config.NumberColumn(format="%+.0f"),
        "Change (%)": st.column_config.NumberColumn(format="%+.0f%%", help="Empty when the previous month had no spending."),
    })


def _insight_cards(report: ins.InsightReport) -> None:
    st.markdown("##### Smart insights")
    if not report.insights:
        if report.overview["comparison_available"]:
            st.success("No significant changes found. Your category spending is steady and nothing exceeded a budget in the latest month.")
        else:
            st.info("Insights about changes need at least two consecutive months of data.")
        return
    for insight in report.insights[:VISIBLE_INSIGHTS]:
        st.markdown(_card(insight), unsafe_allow_html=True)
    if len(report.insights) > VISIBLE_INSIGHTS:
        with st.expander(f"Show {len(report.insights) - VISIBLE_INSIGHTS} more insight(s)"):
            for insight in report.insights[VISIBLE_INSIGHTS:]:
                st.markdown(_card(insight), unsafe_allow_html=True)


def _unusual_section(report: ins.InsightReport) -> None:
    st.markdown("##### Unusual transactions")
    if report.unusual.empty:
        st.info(report.unusual_note if "Not enough" in report.unusual_note else "No unusual transactions were found.")
        return
    st.caption(report.unusual_note + " Review each one; a large purchase can be perfectly normal.")
    shown = report.unusual.assign(date=report.unusual["date"].dt.strftime("%Y-%m-%d"))[
        ["date", "description", "category", "amount", "typical", "reason"]]
    shown.columns = ["Date", "Description", "Category", "Amount (₹)", "Typical (₹)", "Why it was flagged"]
    st.dataframe(shown, hide_index=True, width="stretch", column_config={
        "Amount (₹)": st.column_config.NumberColumn(format="%.0f"), "Typical (₹)": st.column_config.NumberColumn(format="%.0f"),
        "Why it was flagged": st.column_config.TextColumn(width="large")})


def _recurring_section(report: ins.InsightReport) -> None:
    st.markdown("##### Recurring payments")
    rec = report.recurring
    if rec.empty:
        st.info("No recurring payments were detected. A payment needs at least 3 occurrences at regular intervals.")
        return
    fixed = ins.active_fixed(rec)
    m1, m2, m3 = st.columns(3)
    m1.metric("Active fixed payments", len(fixed))
    m2.metric("Estimated per month", inr(fixed["est_monthly"].sum()) if len(fixed) else inr(0))
    m3.metric("Estimated per year", inr(fixed["est_annual"].sum()) if len(fixed) else inr(0))
    st.caption("Detected from repeated payments at regular intervals. Totals include only active fixed-amount payments. "
               "Varying-amount patterns (such as utility bills) are listed but not totalled. These are pattern-based "
               "estimates, so please verify them.")
    shown = rec.assign(last_date=rec["last_date"].dt.strftime("%Y-%m-%d"), next_expected=rec["next_expected"].dt.strftime("%Y-%m-%d"))[
        ["merchant", "category", "cadence", "pattern", "typical_amount", "occurrences", "est_monthly", "est_annual",
         "last_date", "next_expected", "status", "confidence"]]
    shown.columns = ["Payment", "Category", "Repeats", "Pattern", "Typical (₹)", "Seen", "Per month (₹)", "Per year (₹)",
                     "Last paid", "Next expected", "Status", "Confidence"]
    st.dataframe(shown, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.0f") for c in ("Typical (₹)", "Per month (₹)", "Per year (₹)")})


def _ai_section(report: ins.InsightReport) -> None:
    st.markdown("##### Explain with Gemini (optional)")
    st.caption("The cards above are always calculated locally. Gemini can write a short plain-language summary of those "
               "verified results. Any figure it mentions that is not in the results causes its text to be discarded.")
    include = st.checkbox("Include merchant names in what is sent to Gemini", value=False, key="ins_include_merchants",
                          help="Off by default: only categories and calculated figures are sent.")
    facts = ins.facts_text(report, include)
    with st.expander("Inspect exactly what will be sent to Gemini"):
        st.code(build_insights_prompt(facts), language="text")
    if not is_configured():
        st.caption("Gemini is not configured, so the AI summary is unavailable. The calculated insights above are unaffected.")
    fingerprint = str(hash((facts, include)))
    if st.button("Explain these insights with Gemini", disabled=not is_configured() or not report.has_data):
        with st.spinner("Gemini is writing the explanation..."):
            result = explain_insights(report, include_merchants=include)
        st.session_state["insights_ai"] = {"fp": fingerprint, **result}
    saved = st.session_state.get("insights_ai")
    if saved and saved.get("fp") == fingerprint:
        if saved["warning"]:
            st.info(saved["warning"])
        st.markdown(saved["text"])
        st.caption("AI-generated explanation of calculated results." if saved["source"] == "gemini"
                   else "Calculated summary (no AI text was used).")


def render(df: pd.DataFrame, budgets: dict | None = None) -> None:
    st.subheader("5. Smart Spending Insights")
    st.caption("Calculated from your loaded transactions only. Insights are statistical observations, not professional advice.")
    try:
        spend, _ = ins.prepare_transactions(df)
    except ValueError as exc:
        st.error(f"Insights are unavailable: {exc}")
        return
    categories = sorted(c for c in spend["category"].unique() if c != ins.UNATTRIBUTED)
    if spend.empty:
        st.info("There are no expense transactions to analyse yet.")
        return
    dmin, dmax = spend["date"].min().date(), spend["date"].max().date()
    signature = f"{dmin}_{dmax}_{hash(tuple(categories)) & 0xFFFFFF}"  # new dataset => fresh filter widgets
    f1, f2 = st.columns(2)
    date_range = f1.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax, key=f"ins_range_{signature}")
    chosen = f2.multiselect("Categories", categories, default=categories, key=f"ins_cats_{signature}")
    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.info("Pick both a start and an end date.")
        return
    if not chosen:
        st.info("Select at least one category.")
        return
    report = ins.analyze(df, budgets, date_range[0], date_range[1], None if set(chosen) == set(categories) else chosen)

    for note in report.notes:
        st.info(note)
    if not report.has_data:
        return
    _overview(report)
    _category_section(report)
    _insight_cards(report)
    _unusual_section(report)
    _recurring_section(report)
    _ai_section(report)