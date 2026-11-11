"""
Helper functions and visualization utilities for the Environment Classifier
and Demonstrator application.
"""

from typing import Dict, Any, Optional
import plotly.graph_objects as go
import pandas as pd

from config import (
    CLASS_COLD,
    CLASS_COMFORTABLE,
    CLASS_HOT,
    STATUS_COLORS,
    ACTION_OFF,
    ACTION_HEAT,
    ACTION_COOL,
)


def get_status_card_html(classification: str, current_temp: float, desired_temp: float) -> str:
    """
    Generate an HTML status card representing the temperature classification.
    """
    info = STATUS_COLORS.get(classification, STATUS_COLORS[CLASS_COMFORTABLE])
    delta = current_temp - desired_temp
    delta_str = f"{'+' if delta > 0 else ''}{delta:.1f}°C from target ({desired_temp:.1f}°C)"

    return f"""
    <div style="
        background-color: {info['bg_light']};
        border: 2px solid {info['border']};
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    ">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 32px;">{info['icon']}</span>
                <div>
                    <div style="font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: {info['color']};">
                        Environment Status
                    </div>
                    <div style="font-size: 24px; font-weight: 800; color: #1e293b;">
                        {classification.upper()}
                    </div>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 26px; font-weight: 800; color: {info['color']};">
                    {current_temp:.1f}°C
                </div>
                <div style="font-size: 13px; font-weight: 500; color: #64748b;">
                    {delta_str}
                </div>
            </div>
        </div>
        <div style="margin-top: 8px; font-size: 13px; color: #475569; border-top: 1px solid {info['border']}; padding-top: 8px;">
            {info['description']}
        </div>
    </div>
    """


def create_temperature_gauge(
    current_temp: float,
    desired_temp: float,
    comfort_tolerance: float = 1.5,
    min_gauge: float = 0.0,
    max_gauge: float = 45.0,
) -> go.Figure:
    """
    Build an interactive gauge chart for current room temperature,
    with distinct colored bands for Cold, Comfortable, and Hot zones.
    """
    lower_bound = max(min_gauge, desired_temp - comfort_tolerance)
    upper_bound = min(max_gauge, desired_temp + comfort_tolerance)

    if current_temp < lower_bound:
        bar_color = STATUS_COLORS[CLASS_COLD]["color"]
    elif current_temp > upper_bound:
        bar_color = STATUS_COLORS[CLASS_HOT]["color"]
    else:
        bar_color = STATUS_COLORS[CLASS_COMFORTABLE]["color"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=current_temp,
            number={"suffix": "°C", "font": {"size": 36, "color": "#0f172a"}},
            delta={
                "reference": desired_temp,
                "increasing": {"color": "#dc2626"},
                "decreasing": {"color": "#0284c7"},
                "suffix": "° vs Target",
            },
            title={"text": "<b>Current Room Temperature</b>", "font": {"size": 16, "color": "#334155"}},
            gauge={
                "axis": {"range": [min_gauge, max_gauge], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "#f8fafc",
                "borderwidth": 1,
                "bordercolor": "#cbd5e1",
                "steps": [
                    {"range": [min_gauge, lower_bound], "color": "#e0f2fe"},
                    {"range": [lower_bound, upper_bound], "color": "#dcfce7"},
                    {"range": [upper_bound, max_gauge], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#059669", "width": 4},
                    "thickness": 0.8,
                    "value": desired_temp,
                },
            },
        )
    )

    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif"),
    )
    return fig


def create_temperature_history_chart(
    df: pd.DataFrame,
    desired_temp: float,
    comfort_tolerance: float = 1.5,
) -> go.Figure:
    """
    Generate an interactive line chart tracking Room Temperature and Outside Temperature.
    """
    fig = go.Figure()

    if df.empty:
        fig.update_layout(
            title="No simulation data yet.",
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    steps = df["step"]
    lower_bound = desired_temp - comfort_tolerance
    upper_bound = desired_temp + comfort_tolerance

    # Shaded Comfortable Zone
    fig.add_trace(
        go.Scatter(
            x=list(steps) + list(steps[::-1]),
            y=[upper_bound] * len(steps) + [lower_bound] * len(steps),
            fill="toself",
            fillcolor="rgba(34, 197, 94, 0.12)",
            line=dict(color="rgba(34, 197, 94, 0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Comfort Zone",
        )
    )

    # Desired Target Temperature
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=[desired_temp] * len(steps),
            mode="lines",
            line=dict(color="#16a34a", width=1.8, dash="dash"),
            name=f"Target ({desired_temp:.1f}°C)",
            hovertemplate="Target: %{y:.1f}°C<extra></extra>",
        )
    )

    # Outside Ambient Temperature
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=df["outside_temp"],
            mode="lines",
            line=dict(color="#f97316", width=2, dash="dot"),
            name="Outside Temp",
            hovertemplate="Step %{x}: Outside %{y:.1f}°C<extra></extra>",
        )
    )

    # Indoor Room Temperature
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=df["room_temp"],
            mode="lines+markers",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=5, color="#1d4ed8"),
            name="Room Temp",
            hovertemplate="Step %{x}: Room %{y:.2f}°C<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Temperature Trajectory Over Time</b>",
        title_font=dict(size=16, color="#1e293b"),
        xaxis_title="Simulation Time Step",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        height=380,
        margin=dict(l=30, r=20, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.7)",
        ),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 250, 252, 0.6)",
    )

    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")

    return fig


def create_comparison_bar_chart(
    room_temp: float,
    desired_temp: float,
    outside_temp: float,
) -> go.Figure:
    """
    Build a side-by-side comparison bar chart for Indoor, Desired, and Outdoor temperatures.
    """
    categories = ["Current Indoor", "Target (Comfort)", "Outside Ambient"]
    temperatures = [room_temp, desired_temp, outside_temp]
    colors = ["#2563eb", "#16a34a", "#ea580c"]

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=temperatures,
            marker_color=colors,
            text=[f"{t:.1f}°C" for t in temperatures],
            textposition="auto",
            textfont=dict(size=14, color="white", weight="bold"),
            width=0.45,
        )
    )

    fig.update_layout(
        title="<b>Temperature Comparison</b>",
        title_font=dict(size=16, color="#1e293b"),
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        height=260,
        margin=dict(l=25, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 250, 252, 0.6)",
    )
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")

    return fig


def create_dual_comparison_chart(
    df_reflex: pd.DataFrame,
    df_utility: pd.DataFrame,
    desired_temp: float,
    comfort_tolerance: float = 1.5,
) -> go.Figure:
    """
    Generate an interactive comparative trajectory chart plotting both
    the Simple Reflex Agent and the Utility-Based Agent against the comfort zone.
    """
    fig = go.Figure()

    if df_reflex.empty or df_utility.empty:
        fig.update_layout(title="No simulation data yet.")
        return fig

    steps = df_reflex["step"]
    lower_bound = desired_temp - comfort_tolerance
    upper_bound = desired_temp + comfort_tolerance

    # 1. Shaded Comfortable Zone
    fig.add_trace(
        go.Scatter(
            x=list(steps) + list(steps[::-1]),
            y=[upper_bound] * len(steps) + [lower_bound] * len(steps),
            fill="toself",
            fillcolor="rgba(34, 197, 94, 0.12)",
            line=dict(color="rgba(34, 197, 94, 0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Comfort Zone",
        )
    )

    # 2. Desired Target Temperature
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=[desired_temp] * len(steps),
            mode="lines",
            line=dict(color="#16a34a", width=1.8, dash="dash"),
            name=f"Target ({desired_temp:.1f}°C)",
            hovertemplate="Target: %{y:.1f}°C<extra></extra>",
        )
    )

    # 3. Outside Ambient Temperature
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=df_reflex["outside_temp"],
            mode="lines",
            line=dict(color="#94a3b8", width=1.8, dash="dot"),
            name="Outside Ambient",
            hovertemplate="Outside: %{y:.1f}°C<extra></extra>",
        )
    )

    # 4. Simple Reflex Agent Trajectory
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=df_reflex["room_temp"],
            mode="lines+markers",
            line=dict(color="#dc2626", width=2.5),
            marker=dict(size=4, color="#b91c1c"),
            name="Simple Reflex Thermostat",
            hovertemplate="Reflex: %{y:.2f}°C<extra></extra>",
        )
    )

    # 5. Utility-Based Agent Trajectory
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=df_utility["room_temp"],
            mode="lines+markers",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=5, color="#1d4ed8"),
            name="Utility-Based Agent",
            hovertemplate="Utility Agent: %{y:.2f}°C<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Head-to-Head Temperature Trajectory: Reflex vs. Utility Agent</b>",
        title_font=dict(size=16, color="#1e293b"),
        xaxis_title="Simulation Time Step",
        yaxis_title="Room Temperature (°C)",
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=20, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.7)",
        ),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 250, 252, 0.6)",
    )

    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")

    return fig


def create_action_distribution_chart(
    metrics_reflex: Dict[str, Any],
    metrics_utility: Dict[str, Any],
) -> go.Figure:
    """
    Generate a grouped bar chart comparing action distributions (% OFF, % HEAT, % COOL).
    """
    actions = ["OFF (Drift)", "HEAT (Active)", "COOL (Active)"]
    reflex_pcts = [
        metrics_reflex.get("off_percentage", 0.0),
        metrics_reflex.get("heat_percentage", 0.0),
        metrics_reflex.get("cool_percentage", 0.0),
    ]
    utility_pcts = [
        metrics_utility.get("off_percentage", 0.0),
        metrics_utility.get("heat_percentage", 0.0),
        metrics_utility.get("cool_percentage", 0.0),
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Simple Reflex Thermostat",
                x=actions,
                y=reflex_pcts,
                marker_color="#dc2626",
                text=[f"{p:.1f}%" for p in reflex_pcts],
                textposition="auto",
            ),
            go.Bar(
                name="Utility-Based Agent",
                x=actions,
                y=utility_pcts,
                marker_color="#2563eb",
                text=[f"{p:.1f}%" for p in utility_pcts],
                textposition="auto",
            ),
        ]
    )

    fig.update_layout(
        barmode="group",
        title="<b>Action Distribution Comparison (% of Time)</b>",
        title_font=dict(size=15, color="#1e293b"),
        yaxis_title="Percentage (%)",
        template="plotly_white",
        height=300,
        margin=dict(l=25, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 250, 252, 0.6)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")

    return fig


def create_utility_breakdown_chart(utilities: Dict[str, Dict[str, float]]) -> go.Figure:
    """
    Generate a breakdown bar chart showing the utility components for each candidate action.
    """
    actions = list(utilities.keys())
    u_comfort = [utilities[a]["u_comfort"] for a in actions]
    u_energy = [utilities[a]["u_energy"] for a in actions]
    u_switch = [utilities[a]["u_switch"] for a in actions]
    total_u = [utilities[a]["total_utility"] for a in actions]

    fig = go.Figure()

    fig.add_trace(go.Bar(name="Comfort Score", x=actions, y=u_comfort, marker_color="#16a34a"))
    fig.add_trace(go.Bar(name="Energy Cost Penalty", x=actions, y=u_energy, marker_color="#ea580c"))
    fig.add_trace(go.Bar(name="Switching Penalty", x=actions, y=u_switch, marker_color="#9333ea"))
    fig.add_trace(
        go.Scatter(
            name="Net Utility",
            x=actions,
            y=total_u,
            mode="lines+markers+text",
            line=dict(color="#0f172a", width=3),
            marker=dict(size=10, symbol="diamond"),
            text=[f"{u:.2f}" for u in total_u],
            textposition="top center",
        )
    )

    fig.update_layout(
        title="<b>Utility Function Component Breakdown per Action</b>",
        title_font=dict(size=15, color="#1e293b"),
        barmode="relative",
        yaxis_title="Utility Value",
        template="plotly_white",
        height=280,
        margin=dict(l=25, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 250, 252, 0.6)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")

    return fig
