"""Plotly chart builders. They only build figures, so they are easy to test."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

import config
from src.financial_calculations import format_inr

INCOME_COLOR = "#14B8A6"
EXPENSE_COLOR = "#F43F5E"
BUDGET_COLOR = "#6366F1"
PALETTE = ["#6366F1", "#EC4899", "#14B8A6", "#F59E0B", "#8B5CF6", "#F43F5E", "#06B6D4", "#84CC16", "#FB923C", "#3B82F6", "#D946EF", "#64748B"]
RUPEE = config.CURRENCY_SYMBOL


def _base_layout(fig: go.Figure, title: str, height: int = 380, legend_below: bool = True) -> go.Figure:
    """Shared look. The title sits top-left; the legend goes BELOW the plot so they never overlap."""
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16, color="#312e81"), x=0.02, xanchor="left", y=0.96, yanchor="top"),
        height=height,
        margin=dict(l=10, r=10, t=60, b=70 if legend_below else 20),
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5) if legend_below else dict(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Arial, sans-serif", color="#374151"),
        hoverlabel=dict(font_size=13),
    )
    return fig


def _money_axis(fig: go.Figure) -> go.Figure:
    fig.update_yaxes(tickprefix=RUPEE, tickformat=",.0f", gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    return fig


def income_vs_expenses_chart(income: float, expenses: float) -> go.Figure:
    values = [income, expenses]
    fig = go.Figure(
        go.Bar(
            x=["Income", "Expenses"],
            y=values,
            marker_color=[INCOME_COLOR, EXPENSE_COLOR],
            text=[format_inr(v) for v in values],
            textposition="outside",
            customdata=[format_inr(v) for v in values],
            hovertemplate="%{x}: %{customdata}<extra></extra>",
        )
    )
    fig.update_yaxes(rangemode="tozero")
    return _money_axis(_base_layout(fig, "Income vs Expenses (total)"))


def expense_donut_chart(category_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=category_df["category"],
            values=category_df["amount"],
            hole=0.55,
            sort=False,
            marker=dict(colors=PALETTE),
            customdata=[format_inr(v) for v in category_df["amount"]],
            hovertemplate="%{label}<br>%{customdata} (%{percent})<extra></extra>",
            textinfo="percent",
        )
    )
    _base_layout(fig, "Expense breakdown by category", legend_below=False)
    fig.update_layout(legend=dict(orientation="v", x=1.0, xanchor="left", y=0.5, yanchor="middle"), margin=dict(l=10, r=110, t=60, b=20))
    return fig


def monthly_trend_chart(monthly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column, label, color in (("income", "Income", INCOME_COLOR), ("expenses", "Expenses", EXPENSE_COLOR)):
        fig.add_trace(
            go.Scatter(
                x=monthly["month"],
                y=monthly[column],
                name=label,
                mode="lines+markers",
                line=dict(color=color, width=3),
                marker=dict(size=8),
                customdata=[format_inr(v) for v in monthly[column]],
                hovertemplate=f"{label} (%{{x}}): %{{customdata}}<extra></extra>",
            )
        )
    fig.update_xaxes(type="category", title="Month")
    return _money_axis(_base_layout(fig, "Monthly income and expense trend"))


def budget_vs_actual_chart(comparison: pd.DataFrame, period_label: str = "average month") -> go.Figure:
    df = comparison.sort_values("budget", ascending=False)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["category"], y=df["budget"], name="Budget", marker_color=BUDGET_COLOR,
            customdata=[format_inr(v) for v in df["budget"]],
            hovertemplate="%{x}<br>Budget: %{customdata}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["category"], y=df["actual"], name="Actual",
            marker_color=[EXPENSE_COLOR if a > b else INCOME_COLOR for a, b in zip(df["actual"], df["budget"])],
            customdata=[format_inr(v) for v in df["actual"]],
            hovertemplate="%{x}<br>Actual: %{customdata}<extra></extra>",
        )
    )
    fig.update_layout(barmode="group")
    return _money_axis(_base_layout(fig, f"Budget vs actual spending ({period_label})", height=420))


def category_comparison_chart(table: pd.DataFrame, previous_label: str, latest_label: str, top_n: int = 10) -> go.Figure:
    """Grouped bars: each category's spending in the previous month vs the latest month."""
    df = table.assign(_size=table[["previous", "latest"]].max(axis=1)).nlargest(top_n, "_size")
    fig = go.Figure()
    for column, label, color in (("previous", previous_label, "#A5B4FC"), ("latest", latest_label, "#6366F1")):
        fig.add_trace(
            go.Bar(
                x=df["category"], y=df[column], name=label, marker_color=color,
                customdata=[format_inr(v) for v in df[column]],
                hovertemplate=f"%{{x}}<br>{label}: %{{customdata}}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group")
    return _money_axis(_base_layout(fig, f"Spending by category: {previous_label} vs {latest_label}", height=420))