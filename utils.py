"""
Helper functions and visualization utilities for the Environment Classifier
and Demonstrator application (Phase 2: Utility-Based Intelligent Agent).
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
    ACTION_METADATA,
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
        margin-bottom: 16px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    ">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 32px;">{info['icon']}</span>
                <div>
                    <div style="font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: {info['color']};">
                        Environment Status
                    </div>
                    <div style="font-size: 22px; font-weight: 800; color: #1e293b;">
                        {classification.upper()}
                    </div>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 24px; font-weight: 800; color: {info['color']};">
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


def get_decision_card_html(
    current_temp: float,
    desired_temp: float,
    selected_action: str,
    predicted_temp: float,
    utility_score: float,
    reason: str,
) -> str:
    """
    Generate a styled decision explanation card for the Utility-Based Agent.
    """
    meta = ACTION_METADATA.get(selected_action, ACTION_METADATA[ACTION_OFF])

    return f"""
    <div style="
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #cbd5e1;
        border-left: 6px solid {meta['color']};
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div>
                <span style="font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b;">
                    Utility Agent Decision
                </span>
                <div style="font-size: 20px; font-weight: 800; color: #0f172a;">
                    Selected Action: <span style="color: {meta['color']};">{meta['label']}</span>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 12px; color: #64748b; font-weight: 600;">Max Utility Score</div>
                <div style="font-size: 22px; font-weight: 800; color: #0f172a;">{utility_score:.2f}</div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; background: white; padding: 10px 14px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 12px;">
            <div>
                <div style="font-size: 11px; color: #64748b;">Current Temp</div>
                <div style="font-size: 16px; font-weight: 700; color: #1e293b;">{current_temp:.2f}°C</div>
            </div>
            <div>
                <div style="font-size: 11px; color: #64748b;">Desired Target</div>
                <div style="font-size: 16px; font-weight: 700; color: #16a34a;">{desired_temp:.1f}°C</div>
            </div>
            <div>
                <div style="font-size: 11px; color: #64748b;">Predicted Outcome</div>
                <div style="font-size: 16px; font-weight: 700; color: {meta['color']};">{predicted_temp:.2f}°C</div>
            </div>
        </div>

        <div style="font-size: 13.5px; color: #334155; line-height: 1.45;">
            <b>💡 Decision Reason:</b> {reason}
        </div>
    </div>
    """


def create_utility_comparison_chart(
    action_evaluations: Dict[str, Any],
    selected_action: str,
) -> go.Figure:
    """
    Generate a bar chart comparing the total utility scores for HEAT, COOL, and OFF.
    The selected action is highlighted.
    """
    actions = [ACTION_HEAT, ACTION_COOL, ACTION_OFF]
    labels = ["HEAT (+Power)", "COOL (-Power)", "OFF (Drift)"]
    utilities = [action_evaluations[a].utility if a in action_evaluations else 0.0 for a in actions]

    colors = []
    border_colors = []
    border_widths = []

    for a in actions:
        if a == selected_action:
            colors.append("#2563eb")  # Highlight winner
            border_colors.append("#1e3a8a")
            border_widths.append(3)
        else:
            colors.append("#cbd5e1")
            border_colors.append("#94a3b8")
            border_widths.append(1)

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=utilities,
            marker=dict(
                color=colors,
                line=dict(color=border_colors, width=border_widths),
            ),
            text=[f"{u:.2f}" for u in utilities],
            textposition="auto",
            textfont=dict(size=14, weight="bold"),
            width=0.45,
        )
    )

    fig.update_layout(
        title="<b>Action Utility Comparison (Highest Selected)</b>",
        title_font=dict(size=15, color="#1e293b"),
        yaxis_title="Net Utility Score",
        template="plotly_white",
        height=280,
        margin=dict(l=25, r=20, t=45, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 250, 252, 0.6)",
    )
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")

    return fig


def create_utility_factors_breakdown_chart(
    action_evaluations: Dict[str, Any],
) -> go.Figure:
    """
    Generate a grouped/stacked bar chart illustrating the 4 penalty factors:
    Comfort, Energy, Instability, and Overshoot for HEAT, COOL, and OFF.
    """
    actions = [ACTION_HEAT, ACTION_COOL, ACTION_OFF]
    labels = ["HEAT", "COOL", "OFF"]

    comfort_penalties = [action_evaluations[a].weighted_comfort for a in actions]
    energy_costs = [action_evaluations[a].weighted_energy for a in actions]
    instability_penalties = [action_evaluations[a].weighted_stability for a in actions]
    overshoot_penalties = [action_evaluations[a].weighted_overshoot for a in actions]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Comfort Penalty", x=labels, y=comfort_penalties, marker_color="#ef4444"))
    fig.add_trace(go.Bar(name="Energy Cost", x=labels, y=energy_costs, marker_color="#f97316"))
    fig.add_trace(go.Bar(name="Instability Penalty", x=labels, y=instability_penalties, marker_color="#8b5cf6"))
    fig.add_trace(go.Bar(name="Overshoot Penalty", x=labels, y=overshoot_penalties, marker_color="#ec4899"))

    fig.update_layout(
        title="<b>Weighted Penalty Factors Decomposition (Lower is Better)</b>",
        title_font=dict(size=15, color="#1e293b"),
        barmode="stack",
        yaxis_title="Weighted Cost",
        template="plotly_white",
        height=280,
        margin=dict(l=25, r=20, t=45, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 250, 252, 0.6)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")

    return fig


def create_agent_temperature_history_chart(
    df: pd.DataFrame,
    desired_temp: float,
    comfort_tolerance: float = 1.5,
) -> go.Figure:
    """
    Generate an interactive line chart tracking Room Temperature, Predicted Temperature,
    Outside Temperature, and the Comfort Zone across simulation steps.
    """
    fig = go.Figure()

    if df.empty:
        fig.update_layout(title="No simulation steps executed yet.", height=380)
        return fig

    steps = df["step"]
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

    # 2. Desired Target Temperature (Center Line)
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
            y=df["outside_temp"],
            mode="lines",
            line=dict(color="#94a3b8", width=1.8, dash="dot"),
            name="Outside Ambient",
            hovertemplate="Step %{x}: Outside %{y:.1f}°C<extra></extra>",
        )
    )

    # 4. Predicted Temperature (Agent Internal Expectation)
    if "predicted_temp" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=df["predicted_temp"],
                mode="lines",
                line=dict(color="#a855f7", width=1.8, dash="dashdot"),
                name="Agent Predicted Temp",
                hovertemplate="Step %{x}: Predicted %{y:.2f}°C<extra></extra>",
            )
        )

    # 5. Actual Room Temperature
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=df["room_temp"],
            mode="lines+markers",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=5, color="#1d4ed8"),
            name="Actual Room Temp",
            hovertemplate="Step %{x}: Room %{y:.2f}°C (%{text})<extra></extra>",
            text=df["action"] if "action" in df.columns else None,
        )
    )

    fig.update_layout(
        title="<b>Room Temperature Trajectory & Agent Performance</b>",
        title_font=dict(size=16, color="#1e293b"),
        xaxis_title="Simulation Time Step",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        height=390,
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


def create_temperature_gauge(
    current_temp: float,
    desired_temp: float,
    comfort_tolerance: float = 1.5,
    min_gauge: float = 0.0,
    max_gauge: float = 45.0,
) -> go.Figure:
    """
    Build an interactive gauge chart for the current room temperature.
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
            number={"suffix": "°C", "font": {"size": 34, "color": "#0f172a"}},
            delta={
                "reference": desired_temp,
                "increasing": {"color": "#dc2626"},
                "decreasing": {"color": "#0284c7"},
                "suffix": "° vs Target",
            },
            title={"text": "<b>Current Room Temperature</b>", "font": {"size": 15, "color": "#334155"}},
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
        height=250,
        margin=dict(l=25, r=25, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif"),
    )
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
        title_font=dict(size=15, color="#1e293b"),
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        height=250,
        margin=dict(l=25, r=20, t=45, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 250, 252, 0.6)",
    )
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")

    return fig
