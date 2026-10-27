"""Page 1: Financial Dashboard."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from src import budget_analyzer as ba
from src import charts, insights_view
from src import financial_calculations as fc
from src.data_loader import DataValidationError, load_sample_transactions, load_transactions, sample_csv_bytes

inr = fc.format_inr
PLOT_CONFIG = {"displayModeBar": False}


def _budget_key(category: str) -> str:
    return f"budget_{category}"


def set_dataset(df: pd.DataFrame, source: str, is_sample: bool) -> None:
    """Store a dataset in session state and reset budgets for it."""
    for key in [k for k in st.session_state if k.startswith("budget_")]:
        del st.session_state[key]
    st.session_state["transactions"] = df
    st.session_state["data_source"] = source
    st.session_state["budgets"] = dict(config.SAMPLE_BUDGETS) if is_sample else {}
    if is_sample:
        for category, amount in config.SAMPLE_BUDGETS.items():
            st.session_state[_budget_key(category)] = float(amount)


def analyzed_csv(df: pd.DataFrame) -> bytes:
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["signed_amount"] = out["amount"].where(out["type"] == "income", -out["amount"])
    return out.to_csv(index=False).encode("utf-8")


def _data_input_section() -> None:
    st.subheader("1. Load your data")
    left, right = st.columns([2, 1])
    with left:
        uploaded = st.file_uploader(
            "Upload a transaction CSV", type=["csv"],
            help="Required columns: date, description, category, type (income/expense), amount.",
        )
        if uploaded is not None:
            upload_id = f"{uploaded.name}:{uploaded.size}:{getattr(uploaded, 'file_id', '')}"
            if st.session_state.get("loaded_upload_id") != upload_id:
                try:
                    df = load_transactions(uploaded)
                except DataValidationError as exc:
                    st.error(f"Could not use '{uploaded.name}': {exc}")
                else:
                    set_dataset(df, uploaded.name, is_sample=False)
                    st.session_state["loaded_upload_id"] = upload_id
                    st.success(f"Loaded {len(df)} transactions from '{uploaded.name}'.")
    with right:
        st.write("")
        if st.button("Load sample dataset", type="primary", width="stretch"):
            set_dataset(load_sample_transactions(), "Sample dataset (fictional data)", is_sample=True)
            st.session_state.pop("loaded_upload_id", None)
            st.success("Sample dataset loaded.")
        st.download_button(
            "Download sample CSV", sample_csv_bytes(), file_name="sample_transactions.csv",
            mime="text/csv", width="stretch",
        )


def _kpi_section(df: pd.DataFrame) -> None:
    s = fc.summarize_transactions(df)
    st.subheader("2. Financial summary")
    rows = [
        [("Total income", inr(s["total_income"]), None), ("Total expenses", inr(s["total_expenses"]), None),
         ("Net savings", inr(s["net_savings"]), None)],
        [("Savings rate", f"{s['savings_rate_pct']:.1f}%" if s["total_income"] > 0 else "N/A",
          None if s["total_income"] > 0 else "No income in data"),
         ("Avg monthly expenses", inr(s["average_monthly_expenses"]), f"over {s['months_covered']} months"),
         ("Highest spending category",
          s["highest_spending_category"] or "N/A",
          inr(s["highest_spending_amount"]) if s["highest_spending_category"] else None)],
    ]
    for row in rows:
        cols = st.columns(3)
        for col, (label, value, note) in zip(cols, row):
            with col.container():
                st.metric(label, value)
                if note:
                    st.caption(note)
    if s["total_income"] > 0 and s["net_savings"] < 0:
        st.warning("Your expenses are higher than your income in this data (negative net savings).")


def _charts_section(df: pd.DataFrame) -> None:
    st.subheader("3. Charts")
    income, expenses = fc.total_income(df), fc.total_expenses(df)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.income_vs_expenses_chart(income, expenses), width="stretch", config=PLOT_CONFIG)
    with right:
        category_df = fc.category_totals(df, "expense")
        if category_df.empty:
            st.info("No expense transactions to chart.")
        else:
            st.plotly_chart(charts.expense_donut_chart(category_df), width="stretch", config=PLOT_CONFIG)
    st.plotly_chart(charts.monthly_trend_chart(fc.monthly_summary(df)), width="stretch", config=PLOT_CONFIG)


def _budget_section(df: pd.DataFrame) -> None:
    st.subheader("4. Budgets")
    categories = sorted(df.loc[df["type"] == "expense", "category"].unique())
    if not categories:
        st.info("Budgets need at least one expense category.")
        return
    with st.expander("Set monthly budgets (\u20b9 per month; 0 = no budget)", expanded=not st.session_state.get("budgets")):
        cols = st.columns(3)
        for i, category in enumerate(categories):
            key = _budget_key(category)
            st.session_state.setdefault(key, 0.0)
            cols[i % 3].number_input(category, min_value=0.0, step=500.0, format="%.0f", key=key)
    budgets = {c: float(st.session_state.get(_budget_key(c), 0.0)) for c in categories}
    st.session_state["budgets"] = budgets

    months = sorted(df["month"].unique())
    period = st.selectbox(
        "Compare against", [ba.AVERAGE_PERIOD] + months,
        format_func=lambda p: "Average month (all data)" if p == ba.AVERAGE_PERIOD else f"Month {p}",
    )
    comparison = ba.compare_budgets(df, budgets, period)
    if comparison.empty:
        st.info("Set at least one monthly budget above to compare it with your actual spending.")
        return

    over = ba.overspent_categories(comparison)
    if over.empty:
        st.success("All budgeted categories are within budget for this period.")
    for row in over.itertuples():
        st.warning(f"**{row.category}** is over budget by {inr(-row.difference)} "
                   f"({inr(row.actual)} spent vs {inr(row.budget)} budget, {row.percent_used:.0f}% used).")
    label = "average month" if period == ba.AVERAGE_PERIOD else period
    st.plotly_chart(charts.budget_vs_actual_chart(comparison, label), width="stretch", config=PLOT_CONFIG)
    display = comparison.rename(columns={
        "category": "Category", "budget": "Budget (\u20b9)", "actual": "Actual (\u20b9)", "difference": "Remaining (\u20b9)",
        "percent_used": "Used (%)", "status": "Status"})
    st.dataframe(
        display, hide_index=True, width="stretch",
        column_config={
            "Budget (\u20b9)": st.column_config.NumberColumn(format="%.0f"),
            "Actual (\u20b9)": st.column_config.NumberColumn(format="%.0f"),
            "Remaining (\u20b9)": st.column_config.NumberColumn(format="%.0f"),
            "Used (%)": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    st.download_button("Download budget comparison (CSV)", comparison.to_csv(index=False).encode("utf-8"),
                       file_name="budget_comparison.csv", mime="text/csv")


def render() -> None:
    _data_input_section()
    df = st.session_state.get("transactions")
    if df is None:
        st.info("**Get started:** click **Load sample dataset** to explore FinPilot with fictional data, "
                "or upload your own CSV. Your file is processed on this computer.")
        return

    st.caption(f"Data source: **{st.session_state.get('data_source', 'unknown')}**  |  "
               f"{len(df)} transactions  |  {df['date'].min():%d %b %Y} to {df['date'].max():%d %b %Y}")
    with st.expander("Data preview", expanded=False):
        preview = df.drop(columns=["month"]).head(25).copy()
        preview["date"] = preview["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(preview, hide_index=True, width="stretch",
                     column_config={"amount": st.column_config.NumberColumn("amount (\u20b9)", format="%.0f")})
        st.caption(f"Showing the first {len(preview)} of {len(df)} rows.")

    _kpi_section(df)
    _charts_section(df)
    _budget_section(df)
    insights_view.render(df, st.session_state.get("budgets", {}))
    st.divider()
    st.download_button("Download analyzed data (CSV)", analyzed_csv(df), file_name="finpilot_analyzed_transactions.csv",
                       mime="text/csv", help="Cleaned transactions plus month and signed_amount columns.")