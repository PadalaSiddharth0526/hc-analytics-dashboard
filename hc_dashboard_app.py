"""
Recruitment & Headcount Analytics Dashboard
Project 1 - HC Analytics | Python · Dash · Plotly · Pandas · OOP Design · GitHub Copilot
Author: [Your Name]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
from typing import Optional


# ─── DATA LAYER (OOP) ────────────────────────────────────────────────────────

class HCDataLoader:
    """Handles data ingestion, preprocessing, and feature engineering for HC analytics."""

    SENIORITY_BINS = [0, 4000, 6500, 8500, 99999]
    SENIORITY_LABELS = ["Junior", "Mid", "Senior", "Lead"]

    def __init__(self, filepath: str):
        self._filepath = filepath
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> "HCDataLoader":
        df = pd.read_csv(self._filepath)
        df["HireDate"] = pd.to_datetime(df["HireDate"])
        df["HireYear"] = df["HireDate"].dt.year
        df["HireMonth"] = df["HireDate"].dt.to_period("M").astype(str)
        df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)
        df["OfferConversion"] = (
            (df["OfferExtended"] == "Yes") & (df["OfferAccepted"] == "Yes")
        ).astype(int)
        # Derived: seniority band from income
        df["SeniorityBand"] = pd.cut(
            df["MonthlyIncome"],
            bins=self.SENIORITY_BINS,
            labels=self.SENIORITY_LABELS,
        ).astype(str)
        # Derived: open requisition (never reached offer stage)
        df["IsOpenRequisition"] = (
            (df["OfferExtended"] == "No") & (df["OfferAccepted"] == "No")
        ).astype(int)
        self._df = df
        return self

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Call .load() first.")
        return self._df

    @property
    def departments(self) -> list:
        return sorted(self._df["Department"].unique().tolist())

    @property
    def years(self) -> list:
        return sorted(self._df["HireYear"].unique().tolist())

    @property
    def regions(self) -> list:
        return sorted(self._df["Region"].unique().tolist())


class HCMetrics:
    """Computes all KPI metrics and aggregations used across the dashboard."""

    SENIORITY_ORDER = ["Junior", "Mid", "Senior", "Lead"]

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def total_headcount(self) -> int:
        return len(self.df[self.df["Attrition"] == "No"])

    def attrition_rate(self) -> float:
        return round(self.df["AttritionFlag"].mean() * 100, 1)

    def offer_to_join_rate(self) -> float:
        extended = self.df[self.df["OfferExtended"] == "Yes"]
        if len(extended) == 0:
            return 0.0
        return round(extended["OfferConversion"].mean() * 100, 1)

    def avg_days_to_fill(self) -> float:
        return round(self.df["DaysToFill"].mean(), 1)

    def open_roles(self) -> int:
        """Count active open requisitions (candidates not yet at offer stage)."""
        return int(self.df["IsOpenRequisition"].sum())

    def headcount_by_dept(self) -> pd.DataFrame:
        return (
            self.df[self.df["Attrition"] == "No"]
            .groupby("Department")
            .size()
            .reset_index(name="Headcount")
        )

    def attrition_by_dept(self) -> pd.DataFrame:
        return (
            self.df.groupby("Department")["AttritionFlag"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "Left", "count": "Total"})
            .assign(Rate=lambda x: (x["Left"] / x["Total"] * 100).round(1))
        )

    def hiring_trend(self) -> pd.DataFrame:
        return (
            self.df.groupby("HireMonth")
            .size()
            .reset_index(name="Hires")
            .sort_values("HireMonth")
        )

    def offer_funnel(self) -> dict:
        total = len(self.df)
        extended = int(self.df["OfferExtended"].eq("Yes").sum())
        accepted = int(self.df["OfferAccepted"].eq("Yes").sum())
        return {"Screened": total, "Offer Extended": extended, "Offer Accepted": accepted}

    def attrition_by_satisfaction(self) -> pd.DataFrame:
        return (
            self.df.groupby("JobSatisfaction")["AttritionFlag"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "AttritionRate", "count": "Count"})
            .assign(AttritionRate=lambda x: (x["AttritionRate"] * 100).round(1))
        )

    def days_to_fill_by_dept(self) -> pd.DataFrame:
        return (
            self.df.groupby("Department")["DaysToFill"]
            .mean()
            .round(1)
            .reset_index()
            .rename(columns={"DaysToFill": "AvgDaysToFill"})
        )

    def salary_by_dept_gender(self) -> pd.DataFrame:
        """Income distribution for box plot — by department and gender."""
        return self.df[["Department", "Gender", "MonthlyIncome"]].copy()

    def overtime_attrition(self) -> pd.DataFrame:
        """Attrition rate split by overtime flag — key HC insight."""
        return (
            self.df.groupby("OverTime")["AttritionFlag"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "AttritionRate", "count": "Count"})
            .assign(AttritionRate=lambda x: (x["AttritionRate"] * 100).round(1))
        )

    def attrition_by_performance(self) -> pd.DataFrame:
        """Attrition rate by performance rating — identifies flight-risk talent."""
        return (
            self.df.groupby("PerformanceRating")["AttritionFlag"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "AttritionRate", "count": "Count"})
            .assign(AttritionRate=lambda x: (x["AttritionRate"] * 100).round(1))
        )

    def headcount_by_seniority(self) -> pd.DataFrame:
        """Headcount pyramid by derived seniority band."""
        order = self.SENIORITY_ORDER
        df_active = self.df[self.df["Attrition"] == "No"]
        result = (
            df_active.groupby("SeniorityBand")
            .size()
            .reset_index(name="Headcount")
        )
        result["SeniorityBand"] = pd.Categorical(
            result["SeniorityBand"], categories=order, ordered=True
        )
        return result.sort_values("SeniorityBand", ascending=False)

    def headcount_by_region(self) -> pd.DataFrame:
        """Headcount by global region — mirrors multi-office HC view."""
        return (
            self.df[self.df["Attrition"] == "No"]
            .groupby("Region")
            .size()
            .reset_index(name="Headcount")
            .sort_values("Headcount", ascending=False)
        )

    def attrition_by_worklife(self) -> pd.DataFrame:
        """Attrition rate by work-life balance score (1–4)."""
        return (
            self.df.groupby("WorkLifeBalance")["AttritionFlag"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "AttritionRate", "count": "Count"})
            .assign(AttritionRate=lambda x: (x["AttritionRate"] * 100).round(1))
        )


# ─── CHART FACTORY (OOP) ────────────────────────────────────────────────────

class ChartFactory:
    """
    Reusable chart-rendering utilities with consistent HC brand styling.
    All methods return plotly Figure objects.
    Developed with GitHub Copilot for accelerated delivery.
    """

    PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"]
    FONT = "Inter, sans-serif"
    BG = "rgba(0,0,0,0)"
    GRID = "#e8e6e0"

    @classmethod
    def _base_layout(cls, title: str = "") -> dict:
        return dict(
            title=dict(text=title, font=dict(size=13, color="#52514e", family=cls.FONT)),
            font=dict(family=cls.FONT, color="#0b0b0b"),
            paper_bgcolor=cls.BG,
            plot_bgcolor=cls.BG,
            margin=dict(l=10, r=10, t=36, b=10),
            legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
            hoverlabel=dict(bgcolor="white", font_size=12, font_family=cls.FONT),
        )

    @classmethod
    def kpi_card(cls, value, label: str, color: str = "#2a78d6") -> go.Figure:
        fig = go.Figure(go.Indicator(
            mode="number",
            value=float(str(value).replace("%", "")),
            number=dict(
                suffix="%" if "%" in str(value) else "",
                font=dict(size=36, color=color, family=cls.FONT),
            ),
        ))
        layout = cls._base_layout(label)
        layout["margin"] = dict(l=10, r=10, t=30, b=0)
        layout["height"] = 110
        fig.update_layout(**layout)
        return fig

    @classmethod
    def bar_dept(cls, df: pd.DataFrame, x: str, y: str, title: str, color: str = None) -> go.Figure:
        fig = go.Figure(go.Bar(
            x=df[x], y=df[y],
            marker_color=color or cls.PALETTE[0],
            marker_line_width=0,
            text=df[y], textposition="outside",
            textfont=dict(size=11),
        ))
        fig.update_layout(
            **cls._base_layout(title),
            height=240,
            yaxis=dict(showgrid=True, gridcolor=cls.GRID, zeroline=False, showticklabels=False),
            xaxis=dict(showgrid=False),
            bargap=0.35,
        )
        return fig

    @classmethod
    def bar_horizontal(cls, df: pd.DataFrame, x: str, y: str, title: str, color: str = None) -> go.Figure:
        fig = go.Figure(go.Bar(
            x=df[x], y=df[y],
            orientation="h",
            marker_color=color or cls.PALETTE[0],
            marker_line_width=0,
            text=df[x], textposition="outside",
            textfont=dict(size=11),
        ))
        fig.update_layout(
            **cls._base_layout(title),
            height=240,
            xaxis=dict(showgrid=True, gridcolor=cls.GRID, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False),
            bargap=0.35,
        )
        return fig

    @classmethod
    def line_trend(cls, df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        fig = go.Figure(go.Scatter(
            x=df[x], y=df[y],
            mode="lines+markers",
            line=dict(color=cls.PALETTE[0], width=2.5),
            marker=dict(size=6, color=cls.PALETTE[0]),
            fill="tozeroy",
            fillcolor="rgba(42,120,214,0.08)",
        ))
        fig.update_layout(
            **cls._base_layout(title),
            height=240,
            xaxis=dict(showgrid=False, dtick="M3", tickformat="%b %Y", tickangle=-30, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor=cls.GRID, zeroline=False),
        )
        return fig

    @classmethod
    def funnel_chart(cls, data: dict, title: str) -> go.Figure:
        labels = list(data.keys())
        values = list(data.values())
        fig = go.Figure(go.Funnel(
            y=labels, x=values,
            textinfo="value+percent initial",
            marker=dict(color=[cls.PALETTE[0], cls.PALETTE[1], cls.PALETTE[2]]),
            connector=dict(line=dict(color="#e8e6e0", width=1)),
        ))
        fig.update_layout(**cls._base_layout(title), height=240)
        return fig

    @classmethod
    def scatter_satisfaction(cls, df: pd.DataFrame, title: str) -> go.Figure:
        fig = go.Figure(go.Bar(
            x=df["JobSatisfaction"].astype(str),
            y=df["AttritionRate"],
            marker_color=[cls.PALETTE[4], cls.PALETTE[2], cls.PALETTE[1], cls.PALETTE[0]],
            text=df["AttritionRate"].astype(str) + "%",
            textposition="outside",
        ))
        fig.update_layout(
            **cls._base_layout(title),
            height=240,
            xaxis=dict(title="Satisfaction (1=Low, 4=High)", showgrid=False),
            yaxis=dict(showgrid=True, gridcolor=cls.GRID, zeroline=False, showticklabels=False),
        )
        return fig

    @classmethod
    def heatmap_days(cls, df: pd.DataFrame, title: str) -> go.Figure:
        fig = go.Figure(go.Bar(
            x=df["Department"],
            y=df["AvgDaysToFill"],
            marker_color=cls.PALETTE[3],
            text=df["AvgDaysToFill"],
            textposition="outside",
        ))
        fig.update_layout(
            **cls._base_layout(title),
            height=240,
            yaxis=dict(showgrid=True, gridcolor=cls.GRID, zeroline=False, showticklabels=False),
            xaxis=dict(showgrid=False),
            bargap=0.35,
        )
        return fig

    @classmethod
    def box_salary(cls, df: pd.DataFrame, title: str) -> go.Figure:
        """Box plot of monthly income by department, split by gender."""
        colors = {"Male": cls.PALETTE[0], "Female": cls.PALETTE[2]}
        fig = go.Figure()
        for gender, grp in df.groupby("Gender"):
            for dept, sub in grp.groupby("Department"):
                fig.add_trace(go.Box(
                    y=sub["MonthlyIncome"],
                    name=dept,
                    legendgroup=gender,
                    legendgrouptitle_text=gender if dept == df["Department"].unique()[0] else "",
                    marker_color=colors.get(gender, cls.PALETTE[0]),
                    boxmean=True,
                    showlegend=True,
                ))
        fig.update_layout(
            **cls._base_layout(title),
            height=280,
            boxmode="group",
            yaxis=dict(showgrid=True, gridcolor=cls.GRID, zeroline=False, title="Monthly income ($)"),
            xaxis=dict(showgrid=False),
        )
        return fig

    @classmethod
    def bar_overtime_attrition(cls, df: pd.DataFrame, title: str) -> go.Figure:
        """Bar chart: attrition rate split by overtime flag."""
        colors = [cls.PALETTE[1], cls.PALETTE[4]]
        fig = go.Figure(go.Bar(
            x=df["OverTime"],
            y=df["AttritionRate"],
            marker_color=colors,
            text=df["AttritionRate"].astype(str) + "%",
            textposition="outside",
        ))
        fig.update_layout(
            **cls._base_layout(title),
            height=240,
            xaxis=dict(showgrid=False, title="Overtime"),
            yaxis=dict(showgrid=True, gridcolor=cls.GRID, zeroline=False, showticklabels=False),
            bargap=0.5,
        )
        return fig

    @classmethod
    def bar_performance_attrition(cls, df: pd.DataFrame, title: str) -> go.Figure:
        """Attrition rate by performance rating."""
        fig = go.Figure(go.Bar(
            x=df["PerformanceRating"].astype(str),
            y=df["AttritionRate"],
            marker_color=[cls.PALETTE[1], cls.PALETTE[0]],
            text=df["AttritionRate"].astype(str) + "%",
            textposition="outside",
        ))
        fig.update_layout(
            **cls._base_layout(title),
            height=240,
            xaxis=dict(title="Performance rating (3=Excellent, 4=Outstanding)", showgrid=False),
            yaxis=dict(showgrid=True, gridcolor=cls.GRID, zeroline=False, showticklabels=False),
            bargap=0.5,
        )
        return fig

    @classmethod
    def bar_seniority_pyramid(cls, df: pd.DataFrame, title: str) -> go.Figure:
        """Horizontal headcount pyramid by seniority band."""
        colors = [cls.PALETTE[3], cls.PALETTE[0], cls.PALETTE[1], cls.PALETTE[2]]
        fig = go.Figure(go.Bar(
            x=df["Headcount"],
            y=df["SeniorityBand"],
            orientation="h",
            marker_color=colors[: len(df)],
            text=df["Headcount"],
            textposition="outside",
        ))
        fig.update_layout(
            **cls._base_layout(title),
            height=240,
            xaxis=dict(showgrid=True, gridcolor=cls.GRID, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False),
            bargap=0.35,
        )
        return fig

    @classmethod
    def bar_region(cls, df: pd.DataFrame, title: str) -> go.Figure:
        """Headcount by global region."""
        fig = go.Figure(go.Bar(
            x=df["Region"],
            y=df["Headcount"],
            marker_color=cls.PALETTE[3],
            text=df["Headcount"],
            textposition="outside",
        ))
        fig.update_layout(
            **cls._base_layout(title),
            height=240,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor=cls.GRID, zeroline=False, showticklabels=False),
            bargap=0.45,
        )
        return fig

    @classmethod
    def bar_worklife_attrition(cls, df: pd.DataFrame, title: str) -> go.Figure:
        """Attrition rate by work-life balance score."""
        fig = go.Figure(go.Bar(
            x=df["WorkLifeBalance"].astype(str),
            y=df["AttritionRate"],
            marker_color=[cls.PALETTE[4], cls.PALETTE[2], cls.PALETTE[1], cls.PALETTE[0]],
            text=df["AttritionRate"].astype(str) + "%",
            textposition="outside",
        ))
        fig.update_layout(
            **cls._base_layout(title),
            height=240,
            xaxis=dict(title="Work-life balance (1=Bad, 4=Best)", showgrid=False),
            yaxis=dict(showgrid=True, gridcolor=cls.GRID, zeroline=False, showticklabels=False),
        )
        return fig


# ─── APP LAYOUT ─────────────────────────────────────────────────────────────

_BASE_DIR = Path(__file__).parent
loader = HCDataLoader(str(_BASE_DIR / "hc_data.csv")).load()
df_full = loader.df

CARD_STYLE = {
    "background": "white",
    "borderRadius": "12px",
    "border": "0.5px solid #e8e6e0",
    "padding": "12px 16px",
    "boxSizing": "border-box",
}

SECTION_LABEL = {
    "fontSize": "11px",
    "fontWeight": "500",
    "letterSpacing": "0.07em",
    "textTransform": "uppercase",
    "color": "#898781",
    "marginBottom": "12px",
    "marginTop": "24px",
    "fontFamily": "Inter, sans-serif",
}

app = Dash(__name__, title="HC Analytics Dashboard")

app.layout = html.Div(
    style={
        "fontFamily": "Inter, -apple-system, sans-serif",
        "background": "#f5f4ef",
        "minHeight": "100vh",
        "padding": "0",
    },
    children=[

        # ── HEADER ──────────────────────────────────────────────────────────
        html.Div(
            style={
                "background": "white",
                "borderBottom": "0.5px solid #e8e6e0",
                "padding": "16px 32px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
            },
            children=[
                html.Div([
                    html.Div("HC Analytics", style={
                        "fontSize": "18px", "fontWeight": "500", "color": "#0b0b0b"
                    }),
                    html.Div("Recruitment & Headcount Dashboard", style={
                        "fontSize": "13px", "color": "#898781", "marginTop": "2px"
                    }),
                ]),
                html.Div("IBM HR Dataset · 1,470 Employees", style={
                    "fontSize": "12px", "color": "#b4b2a9",
                    "background": "#f5f4ef", "padding": "6px 14px",
                    "borderRadius": "20px", "border": "0.5px solid #e8e6e0",
                }),
            ]
        ),

        # ── FILTERS ─────────────────────────────────────────────────────────
        html.Div(
            style={
                "background": "white",
                "borderBottom": "0.5px solid #e8e6e0",
                "padding": "12px 32px",
                "display": "flex",
                "gap": "20px",
                "alignItems": "center",
            },
            children=[
                html.Span("Filter by", style={"fontSize": "12px", "color": "#898781"}),
                dcc.Dropdown(
                    id="filter-dept",
                    options=[{"label": "All Departments", "value": "All"}] +
                            [{"label": d, "value": d} for d in loader.departments],
                    value="All",
                    clearable=False,
                    style={"width": "220px", "fontSize": "13px"},
                ),
                dcc.Dropdown(
                    id="filter-year",
                    options=[{"label": "All Years", "value": 0}] +
                            [{"label": str(y), "value": y} for y in loader.years],
                    value=0,
                    clearable=False,
                    style={"width": "140px", "fontSize": "13px"},
                ),
                dcc.Dropdown(
                    id="filter-gender",
                    options=[
                        {"label": "All Genders", "value": "All"},
                        {"label": "Male", "value": "Male"},
                        {"label": "Female", "value": "Female"},
                    ],
                    value="All",
                    clearable=False,
                    style={"width": "150px", "fontSize": "13px"},
                ),
                dcc.Dropdown(
                    id="filter-region",
                    options=[{"label": "All Regions", "value": "All"}] +
                            [{"label": r, "value": r} for r in loader.regions],
                    value="All",
                    clearable=False,
                    style={"width": "160px", "fontSize": "13px"},
                ),
            ]
        ),

        # ── MAIN CONTENT ────────────────────────────────────────────────────
        html.Div(
            style={"padding": "24px 32px"},
            children=[

                # KPI CARDS
                html.Div("Key metrics", style=SECTION_LABEL),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(5, 1fr)",
                        "gap": "12px",
                    },
                    children=[
                        html.Div(dcc.Graph(id="kpi-headcount", config={"displayModeBar": False}), style=CARD_STYLE),
                        html.Div(dcc.Graph(id="kpi-attrition", config={"displayModeBar": False}), style=CARD_STYLE),
                        html.Div(dcc.Graph(id="kpi-offer", config={"displayModeBar": False}), style=CARD_STYLE),
                        html.Div(dcc.Graph(id="kpi-days", config={"displayModeBar": False}), style=CARD_STYLE),
                        html.Div(dcc.Graph(id="kpi-open", config={"displayModeBar": False}), style=CARD_STYLE),
                    ]
                ),

                # ROW 1: Headcount & Structure
                html.Div("Headcount & workforce structure", style=SECTION_LABEL),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px"},
                    children=[
                        html.Div(dcc.Graph(id="chart-headcount-dept", config={"displayModeBar": False}), style=CARD_STYLE),
                        html.Div(dcc.Graph(id="chart-seniority-pyramid", config={"displayModeBar": False}), style=CARD_STYLE),
                        html.Div(dcc.Graph(id="chart-region", config={"displayModeBar": False}), style=CARD_STYLE),
                    ]
                ),

                # ROW 2: Attrition deep-dive
                html.Div("Attrition analysis", style=SECTION_LABEL),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "12px"},
                    children=[
                        html.Div(dcc.Graph(id="chart-attrition-dept", config={"displayModeBar": False}), style=CARD_STYLE),
                        html.Div(dcc.Graph(id="chart-attrition-sat", config={"displayModeBar": False}), style=CARD_STYLE),
                        html.Div(dcc.Graph(id="chart-overtime-attrition", config={"displayModeBar": False}), style=CARD_STYLE),
                        html.Div(dcc.Graph(id="chart-worklife-attrition", config={"displayModeBar": False}), style=CARD_STYLE),
                    ]
                ),

                # ROW 3: Performance & Compensation
                html.Div("Performance & compensation", style=SECTION_LABEL),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "12px"},
                    children=[
                        html.Div(dcc.Graph(id="chart-salary-box", config={"displayModeBar": False}), style=CARD_STYLE),
                        html.Div(dcc.Graph(id="chart-perf-attrition", config={"displayModeBar": False}), style=CARD_STYLE),
                    ]
                ),

                # ROW 4: Hiring pipeline
                html.Div("Hiring pipeline & open roles", style=SECTION_LABEL),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "2fr 1fr 1fr", "gap": "12px"},
                    children=[
                        html.Div(dcc.Graph(id="chart-hiring-trend", config={"displayModeBar": False}), style=CARD_STYLE),
                        html.Div(dcc.Graph(id="chart-offer-funnel", config={"displayModeBar": False}), style=CARD_STYLE),
                        html.Div(dcc.Graph(id="chart-days-fill", config={"displayModeBar": False}), style=CARD_STYLE),
                    ]
                ),

                # FOOTER
                html.Div(
                    "Built with Python · Dash · Plotly · Pandas · OOP design · GitHub Copilot",
                    style={
                        "textAlign": "center", "fontSize": "11px",
                        "color": "#b4b2a9", "marginTop": "32px", "paddingBottom": "16px"
                    }
                )
            ]
        )
    ]
)


# ─── CALLBACKS ──────────────────────────────────────────────────────────────

def filter_df(dept: str, year: int, gender: str, region: str) -> pd.DataFrame:
    """Apply all sidebar filters and return a filtered dataframe."""
    df = df_full.copy()
    if dept != "All":
        df = df[df["Department"] == dept]
    if year != 0:
        df = df[df["HireYear"] == year]
    if gender != "All":
        df = df[df["Gender"] == gender]
    if region != "All":
        df = df[df["Region"] == region]
    return df


@app.callback(
    Output("kpi-headcount", "figure"),
    Output("kpi-attrition", "figure"),
    Output("kpi-offer", "figure"),
    Output("kpi-days", "figure"),
    Output("kpi-open", "figure"),
    Output("chart-headcount-dept", "figure"),
    Output("chart-seniority-pyramid", "figure"),
    Output("chart-region", "figure"),
    Output("chart-attrition-dept", "figure"),
    Output("chart-attrition-sat", "figure"),
    Output("chart-overtime-attrition", "figure"),
    Output("chart-worklife-attrition", "figure"),
    Output("chart-salary-box", "figure"),
    Output("chart-perf-attrition", "figure"),
    Output("chart-hiring-trend", "figure"),
    Output("chart-offer-funnel", "figure"),
    Output("chart-days-fill", "figure"),
    Input("filter-dept", "value"),
    Input("filter-year", "value"),
    Input("filter-gender", "value"),
    Input("filter-region", "value"),
)
def update_all(dept, year, gender, region):
    df = filter_df(dept, year, gender, region)
    m = HCMetrics(df)
    cf = ChartFactory()

    # KPIs
    kpi1 = cf.kpi_card(m.total_headcount(), "Active headcount", "#2a78d6")
    kpi2 = cf.kpi_card(f"{m.attrition_rate()}%", "Attrition rate", "#e34948")
    kpi3 = cf.kpi_card(f"{m.offer_to_join_rate()}%", "Offer-to-join rate", "#1baf7a")
    kpi4 = cf.kpi_card(m.avg_days_to_fill(), "Avg days to fill", "#eda100")
    kpi5 = cf.kpi_card(m.open_roles(), "Open requisitions", "#4a3aa7")

    # Headcount & structure
    c_hc_dept    = cf.bar_dept(m.headcount_by_dept(), "Department", "Headcount", "Headcount by department", "#2a78d6")
    c_seniority  = cf.bar_seniority_pyramid(m.headcount_by_seniority(), "Headcount pyramid by seniority")
    c_region     = cf.bar_region(m.headcount_by_region(), "Headcount by global region")

    # Attrition
    c_attr_dept  = cf.bar_dept(m.attrition_by_dept(), "Department", "Rate", "Attrition rate by department (%)", "#e34948")
    c_attr_sat   = cf.scatter_satisfaction(m.attrition_by_satisfaction(), "Attrition by job satisfaction")
    c_overtime   = cf.bar_overtime_attrition(m.overtime_attrition(), "Attrition rate: overtime vs. no overtime")
    c_worklife   = cf.bar_worklife_attrition(m.attrition_by_worklife(), "Attrition by work-life balance")

    # Performance & compensation
    c_salary     = cf.box_salary(m.salary_by_dept_gender(), "Salary distribution by department & gender")
    c_perf       = cf.bar_performance_attrition(m.attrition_by_performance(), "Attrition by performance rating")

    # Hiring pipeline
    c_trend      = cf.line_trend(m.hiring_trend(), "HireMonth", "Hires", "Monthly hiring trend")
    c_funnel     = cf.funnel_chart(m.offer_funnel(), "Offer-to-join funnel")
    c_days       = cf.heatmap_days(m.days_to_fill_by_dept(), "Avg days to fill by department")

    return (
        kpi1, kpi2, kpi3, kpi4, kpi5,
        c_hc_dept, c_seniority, c_region,
        c_attr_dept, c_attr_sat, c_overtime, c_worklife,
        c_salary, c_perf,
        c_trend, c_funnel, c_days,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)
