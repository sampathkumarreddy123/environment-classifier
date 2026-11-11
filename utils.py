"""
Helper functions and visualization utilities for the Environment Classifier
and Demonstrator application (Environment, Utility Agent, and Reflex Thermostat).
"""

import textwrap
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


def _clean_html(html: str) -> str:
    """
    Ensure every line of generated HTML is stripped of leading indentation.
    This guarantees Streamlit's Markdown parser will NEVER treat HTML tags
    as indented code blocks.
    """
    return "\n".join(line.lstrip() for line in html.strip().splitlines())


def get_status_card_html(classification: str, current_temp: float, desired_temp: float, dark_mode: bool = False) -> str:
    """
    Generate an HTML status card representing the temperature classification
    using the Biophilic Eco-Tech design palette.
    """
    info = STATUS_COLORS.get(classification, STATUS_COLORS[CLASS_COMFORTABLE])
    delta = current_temp - desired_temp
    delta_str = f"{'+' if delta > 0 else ''}{delta:.1f}°C from target ({desired_temp:.1f}°C)"

    card_bg = "#192720" if dark_mode else "#ffffff"
    card_border = "#304438" if dark_mode else "#dce5d8"
    text_main = "#e7f0e8" if dark_mode else "#26382d"
    text_sub = "#a1b5a5" if dark_mode else "#718174"
    text_desc = "#a1b5a5" if dark_mode else "#718174"

    html = f"""<div style="background-color: {card_bg}; border: 1px solid {card_border}; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);">
<div style="display: flex; align-items: center; justify-content: space-between;">
<div style="display: flex; align-items: center; gap: 12px;">
<span style="font-size: 28px;">{info['icon']}</span>
<div>
<div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: {info['color']};">Environment Status</div>
<div style="font-size: 20px; font-weight: 800; color: {text_main};">{classification.upper()}</div>
</div>
</div>
<div style="text-align: right;">
<div style="font-size: 22px; font-weight: 800; color: {info['color']};">{current_temp:.1f}°C</div>
<div style="font-size: 12px; font-weight: 500; color: {text_sub};">{delta_str}</div>
</div>
</div>
<div style="margin-top: 8px; font-size: 12.5px; color: {text_desc}; border-top: 1px solid {card_border}; padding-top: 8px;">
{info['description']}
</div>
</div>"""
    return _clean_html(html)


def get_decision_card_html(
    current_temp: float,
    desired_temp: float,
    selected_action: str,
    predicted_temp: float,
    utility_score: float,
    reason: str,
    dark_mode: bool = False,
) -> str:
    """
    Generate a styled decision explanation card for the Utility-Based Agent
    in Biophilic Eco-Tech palette.
    """
    meta = ACTION_METADATA.get(selected_action, ACTION_METADATA[ACTION_OFF])

    card_bg = "#192720" if dark_mode else "#ffffff"
    card_border = "#304438" if dark_mode else "#dce5d8"
    sub_bg = "#22352a" if dark_mode else "#eaf0e5"
    sub_border = "#304438" if dark_mode else "#dce5d8"
    reason_bg = "#22352a" if dark_mode else "#eaf0e5"
    text_main = "#e7f0e8" if dark_mode else "#26382d"
    text_body = "#e7f0e8" if dark_mode else "#26382d"
    text_sub = "#a1b5a5" if dark_mode else "#718174"

    action_accent = meta['color']

    html = f"""<div style="background: {card_bg}; border: 1px solid {card_border}; border-left: 5px solid {action_accent}; border-radius: 10px; padding: 14px 18px; margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
<div>
<div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: {text_sub};">Utility Agent Decision</div>
<div style="font-size: 20px; font-weight: 800; color: {text_main};">Selected Action: <span style="color: {action_accent};">{meta['label']}</span></div>
</div>
<div style="text-align: right;">
<div style="font-size: 11px; color: {text_sub}; font-weight: 600; text-transform: uppercase;">Max Utility Score</div>
<div style="font-size: 22px; font-weight: 800; color: {text_main};">{utility_score:.2f}</div>
</div>
</div>
<div style="display: flex; gap: 12px; background: {sub_bg}; padding: 10px 14px; border-radius: 8px; border: 1px solid {sub_border}; margin-bottom: 12px;">
<div style="flex: 1;">
<div style="font-size: 11px; color: {text_sub}; font-weight: 600;">Current Temp</div>
<div style="font-size: 16px; font-weight: 700; color: {text_main};">{current_temp:.2f}°C</div>
</div>
<div style="flex: 1;">
<div style="font-size: 11px; color: {text_sub}; font-weight: 600;">Desired Target</div>
<div style="font-size: 16px; font-weight: 700; color: {'#75c995' if dark_mode else '#39845a'};">{desired_temp:.1f}°C</div>
</div>
<div style="flex: 1;">
<div style="font-size: 11px; color: {text_sub}; font-weight: 600;">Predicted Outcome</div>
<div style="font-size: 16px; font-weight: 700; color: {action_accent};">{predicted_temp:.2f}°C</div>
</div>
</div>
<div style="font-size: 13.5px; color: {text_body}; line-height: 1.45; padding: 8px 12px; background: {reason_bg}; border-radius: 6px;">
<b>💡 Decision Reason:</b> {reason}
</div>
</div>"""
    return _clean_html(html)


def create_utility_comparison_chart(
    action_evaluations: Dict[str, Any],
    selected_action: str,
    dark_mode: bool = False,
) -> go.Figure:
    """
    Generate a bar chart comparing the total utility scores for HEAT, COOL, and OFF.
    The selected action is highlighted with Forest Green in the Biophilic theme.
    """
    actions = [ACTION_HEAT, ACTION_COOL, ACTION_OFF]
    labels = ["HEAT (+Power)", "COOL (-Power)", "OFF (Drift)"]
    utilities = [action_evaluations[a].utility if a in action_evaluations else 0.0 for a in actions]

    winner_color = "#8fd0a8" if dark_mode else "#245c43"
    winner_border = "#9dba91" if dark_mode else "#829b78"
    other_color = "#22352a" if dark_mode else "#dce5d8"
    other_border = "#304438" if dark_mode else "#b9d7c1"

    colors = []
    border_colors = []
    border_widths = []

    for a in actions:
        if a == selected_action:
            colors.append(winner_color)
            border_colors.append(winner_border)
            border_widths.append(2.5)
        else:
            colors.append(other_color)
            border_colors.append(other_border)
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
            textfont=dict(size=14, weight="bold", color="#e7f0e8" if dark_mode else "#26382d"),
            width=0.45,
        )
    )

    fig.update_layout(
        title="<b>Action Utility Comparison (Highest Selected)</b>",
        title_font=dict(size=15, color="#e7f0e8" if dark_mode else "#26382d"),
        yaxis_title="Net Utility Score",
        template="plotly_dark" if dark_mode else "plotly_white",
        height=280,
        margin=dict(l=25, r=20, t=45, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(25, 39, 32, 0.6)" if dark_mode else "rgba(245, 247, 240, 0.6)",
    )
    fig.update_yaxes(showgrid=True, gridcolor="#304438" if dark_mode else "#dce5d8")

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
    desired_temp: float = 22.0,
    comfort_tolerance: float = 1.5,
    dark_mode: bool = False,
    **kwargs: Any,
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
            fillcolor="rgba(117, 201, 149, 0.15)" if dark_mode else "rgba(57, 132, 90, 0.12)",
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
            line=dict(color="#75c995" if dark_mode else "#39845a", width=1.8, dash="dash"),
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
            line=dict(color="#a1b5a5" if dark_mode else "#718174", width=1.8, dash="dot"),
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
                line=dict(color="#9dba91" if dark_mode else "#829b78", width=1.8, dash="dashdot"),
                name="Agent Predicted Temp",
                hovertemplate="Step %{x}: Predicted %{y:.2f}°C<extra></extra>",
            )
        )

    # 5. Actual Room Temperature
    actual_temp_series = df["actual_temp"] if "actual_temp" in df.columns else df.get("room_temp", [0.0] * len(df))
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=actual_temp_series,
            mode="lines+markers",
            line=dict(color="#8fd0a8" if dark_mode else "#245c43", width=3),
            marker=dict(size=5, color="#75c995" if dark_mode else "#1d4a36"),
            name="Actual Room Temp",
            hovertemplate="Step %{x}: Room %{y:.2f}°C (%{text})<extra></extra>",
            text=df["action"] if "action" in df.columns else None,
        )
    )

    fig.update_layout(
        title="<b>Room Temperature Trajectory & Agent Performance</b>",
        title_font=dict(size=16, color="#e7f0e8" if dark_mode else "#26382d"),
        xaxis_title="Simulation Time Step",
        yaxis_title="Temperature (°C)",
        template="plotly_dark" if dark_mode else "plotly_white",
        height=390,
        margin=dict(l=30, r=20, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(25, 39, 32, 0.85)" if dark_mode else "rgba(255,255,255,0.7)",
            font=dict(color="#e7f0e8" if dark_mode else "#26382d"),
        ),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(25, 39, 32, 0.6)" if dark_mode else "rgba(245, 247, 240, 0.6)",
    )

    fig.update_xaxes(showgrid=True, gridcolor="#304438" if dark_mode else "#dce5d8")
    fig.update_yaxes(showgrid=True, gridcolor="#304438" if dark_mode else "#dce5d8")

    return fig


def create_temperature_gauge(
    current_temp: float,
    desired_temp: float,
    comfort_tolerance: float = 1.5,
    min_gauge: float = 0.0,
    max_gauge: float = 45.0,
    dark_mode: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """
    Build an interactive scientific gauge chart for the current room temperature
    using Biophilic Eco-Tech palette.
    """
    lower_bound = max(min_gauge, desired_temp - comfort_tolerance)
    upper_bound = min(max_gauge, desired_temp + comfort_tolerance)

    if current_temp < lower_bound:
        bar_color = "#8fd0a8" if dark_mode else STATUS_COLORS[CLASS_COLD]["color"]
    elif current_temp > upper_bound:
        bar_color = "#e58b80" if dark_mode else STATUS_COLORS[CLASS_HOT]["color"]
    else:
        bar_color = "#75c995" if dark_mode else STATUS_COLORS[CLASS_COMFORTABLE]["color"]

    num_color = "#e7f0e8" if dark_mode else "#26382d"
    title_color = "#a1b5a5" if dark_mode else "#718174"
    step_cold = "rgba(46, 139, 139, 0.15)" if dark_mode else "#eaf0e5"
    step_comfort = "rgba(117, 201, 149, 0.18)" if dark_mode else "#dce5d8"
    step_hot = "rgba(229, 139, 128, 0.15)" if dark_mode else "#fdf2f0"
    border_col = "#304438" if dark_mode else "#dce5d8"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=current_temp,
            number={"suffix": "°C", "font": {"size": 32, "color": num_color}},
            delta={
                "reference": desired_temp,
                "increasing": {"color": "#e58b80" if dark_mode else "#c65b52"},
                "decreasing": {"color": "#8fd0a8" if dark_mode else "#2e8b8b"},
                "suffix": "° vs Target",
            },
            title={"text": "<b>Current Room Temperature</b>", "font": {"size": 14, "color": title_color}},
            gauge={
                "axis": {"range": [min_gauge, max_gauge], "tickwidth": 1, "tickcolor": "#a1b5a5" if dark_mode else "#718174"},
                "bar": {"color": bar_color, "thickness": 0.32},
                "bgcolor": "#192720" if dark_mode else "#ffffff",
                "borderwidth": 1,
                "bordercolor": border_col,
                "steps": [
                    {"range": [min_gauge, lower_bound], "color": step_cold},
                    {"range": [lower_bound, upper_bound], "color": step_comfort},
                    {"range": [upper_bound, max_gauge], "color": step_hot},
                ],
                "threshold": {
                    "line": {"color": "#75c995" if dark_mode else "#39845a", "width": 3.5},
                    "thickness": 0.8,
                    "value": desired_temp,
                },
            },
        )
    )

    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=35, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif"),
    )
    return fig


def create_comparison_bar_chart(
    room_temp: float,
    desired_temp: float,
    outside_temp: float,
    dark_mode: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """
    Build a side-by-side comparison bar chart for Indoor, Desired, and Outdoor temperatures
    in Biophilic palette.
    """
    categories = ["Current Indoor", "Target (Comfort)", "Outside Ambient"]
    temperatures = [room_temp, desired_temp, outside_temp]
    colors = (
        ["#8fd0a8", "#75c995", "#9dba91"]
        if dark_mode
        else ["#245c43", "#39845a", "#829b78"]
    )

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=temperatures,
            marker_color=colors,
            text=[f"{t:.1f}°C" for t in temperatures],
            textposition="auto",
            textfont=dict(size=13, color="white", weight="bold"),
            width=0.42,
        )
    )

    fig.update_layout(
        title="<b>Temperature Comparison</b>",
        title_font=dict(size=15, color="#e7f0e8" if dark_mode else "#26382d"),
        yaxis_title="Temperature (°C)",
        template="plotly_dark" if dark_mode else "plotly_white",
        height=240,
        margin=dict(l=25, r=20, t=45, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(25, 39, 32, 0.6)" if dark_mode else "rgba(245, 247, 240, 0.6)",
    )
    fig.update_yaxes(showgrid=True, gridcolor="#304438" if dark_mode else "#dce5d8")

    return fig


def get_reflex_rules_html(
    current_temp: float,
    lower_threshold: float,
    upper_threshold: float,
    selected_action: str,
    dark_mode: bool = False,
) -> str:
    """
    Generate visual HTML condition-action rule cards using the actual configured
    lower and upper thresholds in Biophilic Eco-Tech palette.
    """
    rules = [
        {
            "action": ACTION_HEAT,
            "condition": f"&lt; {lower_threshold:.1f}°C",
            "arrow_label": "🔥 HEAT",
            "desc": "Room is below lower threshold — turn on heating.",
            "accent": "#e58b80" if dark_mode else "#c65b52",
            "bg_active": "#22352a" if dark_mode else "#fdf2f0",
            "border_active": "#e58b80" if dark_mode else "#c65b52",
        },
        {
            "action": ACTION_OFF,
            "condition": f"{lower_threshold:.1f}°C – {upper_threshold:.1f}°C",
            "arrow_label": "⏸️ OFF",
            "desc": "Room is between thresholds — turn off both systems.",
            "accent": "#75c995" if dark_mode else "#39845a",
            "bg_active": "#22352a" if dark_mode else "#eaf0e5",
            "border_active": "#75c995" if dark_mode else "#39845a",
        },
        {
            "action": ACTION_COOL,
            "condition": f"&gt; {upper_threshold:.1f}°C",
            "arrow_label": "❄️ COOL",
            "desc": "Room is above upper threshold — turn on cooling.",
            "accent": "#8fd0a8" if dark_mode else "#2e8b8b",
            "bg_active": "#22352a" if dark_mode else "#eaf0e5",
            "border_active": "#8fd0a8" if dark_mode else "#2e8b8b",
        },
    ]

    cards_html = []
    for rule in rules:
        is_active = rule["action"] == selected_action
        bg = rule["bg_active"] if is_active else ("#192720" if dark_mode else "#ffffff")
        border = f"2px solid {rule['border_active']}" if is_active else ("1px solid #304438" if dark_mode else "1px solid #dce5d8")
        opacity = "1.0" if is_active else "0.82"
        shadow = "0 3px 8px rgba(0, 0, 0, 0.12)" if is_active else "none"
        badge_bg = rule["accent"] if is_active else ("#304438" if dark_mode else "#829b78")
        badge_text = "ACTIVE RULE" if is_active else "INACTIVE"
        text_head = "#e7f0e8" if dark_mode else "#26382d"
        text_sub = "#a1b5a5" if dark_mode else "#718174"

        cards_html.append(
            f"""<div style="flex: 1; min-width: 210px; background-color: {bg}; border: {border}; border-radius: 10px; padding: 14px 16px; opacity: {opacity}; box-shadow: {shadow};">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
<span style="font-size: 11px; font-weight: 700; letter-spacing: 0.05em; color: white; background-color: {badge_bg}; padding: 2px 8px; border-radius: 999px;">{badge_text}</span>
<span style="font-size: 12px; font-weight: 600; color: {text_sub};">Current: {current_temp:.2f}°C</span>
</div>
<div style="font-size: 18px; font-weight: 800; color: {text_head}; margin-bottom: 4px;">
{rule['condition']} <span style="color: {text_sub}; margin: 0 4px;">→</span> <span style="color: {rule['accent']};">{rule['arrow_label']}</span>
</div>
<div style="font-size: 12.5px; color: {text_sub}; line-height: 1.4;">{rule['desc']}</div>
</div>"""
        )

    return _clean_html(
        '<div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px;">'
        + "".join(cards_html)
        + "</div>"
    )


def get_reflex_decision_card_html(
    current_temp: float,
    lower_threshold: float,
    upper_threshold: float,
    selected_action: str,
    classification: str,
    step_energy: float,
    total_energy: float,
    rule_triggered: str,
    reason: str,
    last_step_action: Optional[str] = None,
    last_step_num: int = 0,
    dark_mode: bool = False,
) -> str:
    """
    Generate a styled status and decision card for the Simple Reflex Thermostat
    in Biophilic Eco-Tech palette.
    """
    meta = ACTION_METADATA.get(selected_action, ACTION_METADATA[ACTION_OFF])
    class_info = STATUS_COLORS.get(classification, STATUS_COLORS[CLASS_COMFORTABLE])

    card_bg = "#192720" if dark_mode else "#ffffff"
    card_border = "#304438" if dark_mode else "#dce5d8"
    sub_bg = "#22352a" if dark_mode else "#eaf0e5"
    sub_border = "#304438" if dark_mode else "#dce5d8"
    reason_bg = "#22352a" if dark_mode else "#eaf0e5"
    text_main = "#e7f0e8" if dark_mode else "#26382d"
    text_sub = "#a1b5a5" if dark_mode else "#718174"

    last_step_note = ""
    if last_step_num > 0 and last_step_action:
        last_meta = ACTION_METADATA.get(last_step_action, ACTION_METADATA[ACTION_OFF])
        last_step_note = (
            f'<div style="margin-top: 8px; font-size: 12.5px; color: {text_sub};">'
            f'<b>Last Executed Step (Step {last_step_num}):</b> '
            f'<span style="color: {last_meta["color"]}; font-weight: 700;">{last_meta["label"]}</span> '
            f'(Step energy: {step_energy:.1f} units)'
            f"</div>"
        )

    html = f"""<div style="background: {card_bg}; border: 1px solid {card_border}; border-left: 5px solid {meta['color']}; border-radius: 10px; padding: 14px 18px; margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
<div>
<div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: {text_sub};">Simple Reflex Thermostat Decision (Current Percept)</div>
<div style="font-size: 20px; font-weight: 800; color: {text_main};">Selected Action: <span style="color: {meta['color']};">{meta['label']}</span></div>
</div>
<div style="text-align: right;">
<div style="font-size: 11px; color: {text_sub}; font-weight: 600; text-transform: uppercase;">Triggered Rule</div>
<div style="font-size: 16px; font-weight: 800; color: {text_main};">{rule_triggered}</div>
</div>
</div>
<div style="display: flex; gap: 12px; flex-wrap: wrap; background: {sub_bg}; padding: 10px 14px; border-radius: 8px; border: 1px solid {sub_border}; margin-bottom: 12px;">
<div style="flex: 1; min-width: 110px;">
<div style="font-size: 11px; color: {text_sub}; font-weight: 600;">Room Temp</div>
<div style="font-size: 16px; font-weight: 700; color: {text_main};">{current_temp:.2f}°C</div>
</div>
<div style="flex: 1; min-width: 110px;">
<div style="font-size: 11px; color: {text_sub}; font-weight: 600;">Lower Threshold</div>
<div style="font-size: 16px; font-weight: 700; color: {'#e58b80' if dark_mode else '#c65b52'};">{lower_threshold:.1f}°C</div>
</div>
<div style="flex: 1; min-width: 110px;">
<div style="font-size: 11px; color: {text_sub}; font-weight: 600;">Upper Threshold</div>
<div style="font-size: 16px; font-weight: 700; color: {'#8fd0a8' if dark_mode else '#2e8b8b'};">{upper_threshold:.1f}°C</div>
</div>
<div style="flex: 1; min-width: 110px;">
<div style="font-size: 11px; color: {text_sub}; font-weight: 600;">Classification</div>
<div style="font-size: 16px; font-weight: 700; color: {class_info['color']};">{class_info['icon']} {classification}</div>
</div>
<div style="flex: 1; min-width: 110px;">
<div style="font-size: 11px; color: {text_sub}; font-weight: 600;">Energy Used</div>
<div style="font-size: 16px; font-weight: 700; color: {text_main};">{total_energy:.1f} units</div>
</div>
</div>
<div style="font-size: 13.5px; color: {text_main}; line-height: 1.45; padding: 8px 12px; background: {reason_bg}; border-radius: 6px;">
<b>⚡ Condition-Action Rule Match:</b> {reason}
{last_step_note}
</div>
</div>"""
    return _clean_html(html)


def create_reflex_temperature_history_chart(
    df: pd.DataFrame,
    lower_threshold: float = 20.0,
    upper_threshold: float = 26.0,
    desired_temp: float = 22.0,
    dark_mode: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """
    Generate an interactive line chart tracking room temperature against the
    configured lower and upper reflex thresholds in Biophilic Eco-Tech palette.
    """
    fig = go.Figure()

    if df.empty:
        fig.update_layout(title="No simulation steps executed yet.", height=380)
        return fig

    steps = df["step"]
    temp_col = "actual_temp" if "actual_temp" in df.columns else "room_temp"
    action_colors = {
        ACTION_HEAT: "#e58b80" if dark_mode else "#c65b52",
        ACTION_COOL: "#8fd0a8" if dark_mode else "#2e8b8b",
        ACTION_OFF: "#75c995" if dark_mode else "#39845a",
    }
    marker_colors = [
        action_colors.get(a, "#a1b5a5" if dark_mode else "#718174")
        for a in (df["action"] if "action" in df.columns else [ACTION_OFF] * len(df))
    ]

    # 1. Shaded OFF Deadband Zone between lower and upper thresholds
    fig.add_trace(
        go.Scatter(
            x=list(steps) + list(steps[::-1]),
            y=[upper_threshold] * len(steps) + [lower_threshold] * len(steps),
            fill="toself",
            fillcolor="rgba(157, 186, 145, 0.16)" if dark_mode else "rgba(130, 155, 120, 0.14)",
            line=dict(color="rgba(34, 197, 94, 0)"),
            hoverinfo="skip",
            showlegend=True,
            name=f"OFF Band ({lower_threshold:.1f}–{upper_threshold:.1f}°C)",
        )
    )

    # 2. Upper Threshold Line (> COOL)
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=[upper_threshold] * len(steps),
            mode="lines",
            line=dict(color="#5bb5b5" if dark_mode else "#2e8b8b", width=1.8, dash="dash"),
            name=f"Upper Threshold ({upper_threshold:.1f}°C → COOL)",
            hovertemplate="Upper Threshold: %{y:.1f}°C<extra></extra>",
        )
    )

    # 3. Lower Threshold Line (< HEAT)
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=[lower_threshold] * len(steps),
            mode="lines",
            line=dict(color="#e58b80" if dark_mode else "#c65b52", width=1.8, dash="dash"),
            name=f"Lower Threshold ({lower_threshold:.1f}°C → HEAT)",
            hovertemplate="Lower Threshold: %{y:.1f}°C<extra></extra>",
        )
    )

    # 4. Desired Target Temperature (Reference)
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=[desired_temp] * len(steps),
            mode="lines",
            line=dict(color="#75c995" if dark_mode else "#39845a", width=1.3, dash="dot"),
            name=f"Desired Target ({desired_temp:.1f}°C)",
            hovertemplate="Target: %{y:.1f}°C<extra></extra>",
        )
    )

    # 5. Outside Ambient Temperature
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=df["outside_temp"],
            mode="lines",
            line=dict(color="#a1b5a5" if dark_mode else "#718174", width=1.6, dash="dot"),
            name="Outside Ambient",
            hovertemplate="Step %{x}: Outside %{y:.1f}°C<extra></extra>",
        )
    )

    # 6. Actual Room Temperature colored by action selected
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=df[temp_col],
            mode="lines+markers",
            line=dict(color="#e7f0e8" if dark_mode else "#26382d", width=2.5),
            marker=dict(size=7, color=marker_colors, line=dict(width=1, color="white")),
            name="Room Temperature",
            text=df["action"] if "action" in df.columns else None,
            hovertemplate="Step %{x}: Room %{y:.2f}°C (Action: %{text})<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Simple Reflex Thermostat Temperature Trajectory</b>",
        title_font=dict(size=16, color="#e7f0e8" if dark_mode else "#26382d"),
        xaxis_title="Simulation Time Step",
        yaxis_title="Temperature (°C)",
        template="plotly_dark" if dark_mode else "plotly_white",
        height=390,
        margin=dict(l=30, r=20, t=55, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(25, 39, 32, 0.85)" if dark_mode else "rgba(255,255,255,0.7)",
            font=dict(color="#e7f0e8" if dark_mode else "#26382d"),
        ),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(25, 39, 32, 0.6)" if dark_mode else "rgba(245, 247, 240, 0.6)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#304438" if dark_mode else "#dce5d8")
    fig.update_yaxes(showgrid=True, gridcolor="#304438" if dark_mode else "#dce5d8")

    return fig


def create_reflex_action_summary_chart(df: pd.DataFrame, dark_mode: bool = False, **kwargs: Any) -> go.Figure:
    """
    Generate a bar chart showing the count of HEAT, COOL, and OFF actions
    taken by the Simple Reflex Thermostat during the simulation.
    """
    active_df = df[df["step"] > 0] if "step" in df.columns and len(df) > 1 else pd.DataFrame()
    actions = [ACTION_HEAT, ACTION_OFF, ACTION_COOL]
    labels = ["🔥 HEAT", "⏸️ OFF", "❄️ COOL"]
    colors = [
        ACTION_METADATA[ACTION_HEAT]["color"],
        ACTION_METADATA[ACTION_OFF]["color"],
        ACTION_METADATA[ACTION_COOL]["color"],
    ]

    if active_df.empty or "action" not in active_df.columns:
        counts = [0, 0, 0]
    else:
        counts = [int((active_df["action"] == a).sum()) for a in actions]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=counts,
            marker_color=colors,
            text=[f"{c} steps" for c in counts],
            textposition="auto",
            textfont=dict(size=13, color="white", weight="bold"),
            width=0.45,
        )
    )

    fig.update_layout(
        title="<b>Reflex Action Distribution</b>",
        title_font=dict(size=15, color="#e7f0e8" if dark_mode else "#26382d"),
        yaxis_title="Number of Steps",
        template="plotly_dark" if dark_mode else "plotly_white",
        height=250,
        margin=dict(l=25, r=20, t=45, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(25, 39, 32, 0.6)" if dark_mode else "rgba(245, 247, 240, 0.6)",
    )
    fig.update_yaxes(showgrid=True, gridcolor="#304438" if dark_mode else "#dce5d8", dtick=1 if max(counts, default=0) <= 10 else None)

    return fig

    return fig


def get_comparison_summary_html(summary: Dict[str, Any], dark_mode: bool = False) -> str:
    """
    Generate an overall comparison summary banner driven strictly by actual
    simulation metric wins (never hardcoded) in Biophilic Eco-Tech palette.
    """
    winner = summary.get("overall_winner", "Tie")
    u_wins = summary.get("utility_wins", 0)
    r_wins = summary.get("reflex_wins", 0)
    ties = summary.get("ties", 0)
    headline = summary.get("headline", "")

    if winner == "Utility-Based Agent":
        accent = "#8fd0a8" if dark_mode else "#245c43"
        bg = "#192720" if dark_mode else "#eaf0e5"
        border = "#8fd0a8" if dark_mode else "#829b78"
        badge = "🌱 UTILITY-BASED AGENT LEADS"
    elif winner == "Simple Reflex Agent":
        accent = "#5bb5b5" if dark_mode else "#2e8b8b"
        bg = "#192720" if dark_mode else "#eaf0e5"
        border = "#5bb5b5" if dark_mode else "#2e8b8b"
        badge = "🔁 SIMPLE REFLEX THERMOSTAT LEADS"
    else:
        accent = "#a1b5a5" if dark_mode else "#718174"
        bg = "#192720" if dark_mode else "#f5f7f0"
        border = "#304438" if dark_mode else "#dce5d8"
        badge = "⚖️ BALANCED / EQUAL PERFORMANCE"

    card_bg = "#192720" if dark_mode else "#ffffff"
    box_bg = "#22352a" if dark_mode else "#eaf0e5"
    box_border = "#304438" if dark_mode else "#dce5d8"
    text_main = "#e7f0e8" if dark_mode else "#26382d"
    text_sub = "#a1b5a5" if dark_mode else "#718174"

    html = f"""<div style="background-color: {card_bg}; border: 2px solid {border}; border-radius: 12px; padding: 16px 20px; margin-bottom: 18px; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
<div>
<div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: {accent};">Overall Comparison Summary (Data-Driven)</div>
<div style="font-size: 22px; font-weight: 800; color: {text_main}; margin-top: 2px;">{badge}</div>
<div style="font-size: 14px; color: {text_sub}; margin-top: 4px;">{headline}</div>
</div>
<div style="display: flex; gap: 12px; flex-wrap: wrap;">
<div style="background: {box_bg}; border: 1px solid {box_border}; border-radius: 8px; padding: 8px 14px; text-align: center;">
<div style="font-size: 11px; font-weight: 700; color: {'#8fd0a8' if dark_mode else '#245c43'};">UTILITY WINS</div>
<div style="font-size: 20px; font-weight: 800; color: {text_main};">{u_wins}</div>
</div>
<div style="background: {box_bg}; border: 1px solid {box_border}; border-radius: 8px; padding: 8px 14px; text-align: center;">
<div style="font-size: 11px; font-weight: 700; color: {'#5bb5b5' if dark_mode else '#2e8b8b'};">REFLEX WINS</div>
<div style="font-size: 20px; font-weight: 800; color: {text_main};">{r_wins}</div>
</div>
<div style="background: {box_bg}; border: 1px solid {box_border}; border-radius: 8px; padding: 8px 14px; text-align: center;">
<div style="font-size: 11px; font-weight: 700; color: #a1b5a5;">TIES</div>
<div style="font-size: 20px; font-weight: 800; color: {text_main};">{ties}</div>
</div>
</div>
</div>
</div>"""
    return _clean_html(html)


def create_comparison_temperature_chart(
    utility_df: Optional[pd.DataFrame] = None,
    reflex_df: Optional[pd.DataFrame] = None,
    desired_temp: float = 22.0,
    comfort_tolerance: float = 1.5,
    df: Optional[pd.DataFrame] = None,
    dark_mode: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """
    Chart 1: Compare Utility-Based Agent and Simple Reflex Agent room temperatures
    alongside the desired temperature reference line and comfort band.
    Biophilic Eco-Tech Colors:
    - Utility-Based Agent: Forest Green (#245C43 Light / #8FD0A8 Dark)
    - Simple Reflex Agent: Muted Teal (#2E8B8B Light / #5BB5B5 Dark)
    - Desired Target: Dashed Natural Green (#39845A Light / #75C995 Dark)
    """
    if utility_df is None and df is not None:
        utility_df = pd.DataFrame({
            "step": df["step"],
            "actual_temp": df.get("utility_temp", df.get("actual_temp", 0.0)),
            "outside_temp": df.get("outside_temp", df.get("utility_outside_temp", 0.0)),
            "action": df.get("utility_action", df.get("action", "")),
        })
        reflex_df = pd.DataFrame({
            "step": df["step"],
            "actual_temp": df.get("reflex_temp", df.get("actual_temp", 0.0)),
            "outside_temp": df.get("outside_temp", df.get("reflex_outside_temp", 0.0)),
            "action": df.get("reflex_action", df.get("action", "")),
        })

    fig = go.Figure()
    if utility_df is None or reflex_df is None or utility_df.empty or reflex_df.empty:
        fig.update_layout(title="No synchronized comparison data yet.", height=380)
        return fig

    steps = utility_df["step"]
    lower_bound = desired_temp - comfort_tolerance
    upper_bound = desired_temp + comfort_tolerance

    # Shaded Comfort Range
    fig.add_trace(
        go.Scatter(
            x=list(steps) + list(steps[::-1]),
            y=[upper_bound] * len(steps) + [lower_bound] * len(steps),
            fill="toself",
            fillcolor="rgba(117, 201, 149, 0.15)" if dark_mode else "rgba(57, 132, 90, 0.12)",
            line=dict(color="rgba(34, 197, 94, 0)"),
            hoverinfo="skip",
            showlegend=True,
            name=f"Comfort Zone ({lower_bound:.1f}–{upper_bound:.1f}°C)",
        )
    )

    # Desired Temperature Reference Line (Natural Green Dashed)
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=[desired_temp] * len(steps),
            mode="lines",
            line=dict(color="#75c995" if dark_mode else "#39845a", width=2.0, dash="dash"),
            name=f"Desired Temp ({desired_temp:.1f}°C)",
            hovertemplate="Target: %{y:.1f}°C<extra></extra>",
        )
    )

    # Outside Ambient Temperature
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=utility_df["outside_temp"],
            mode="lines",
            line=dict(color="#a1b5a5" if dark_mode else "#718174", width=1.5, dash="dot"),
            name="Outside Ambient",
            hovertemplate="Step %{x}: Outside %{y:.1f}°C<extra></extra>",
        )
    )

    # Utility-Based Agent Temperature (Forest Green)
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=utility_df["actual_temp"],
            mode="lines+markers",
            line=dict(color="#8fd0a8" if dark_mode else "#245c43", width=3),
            marker=dict(size=5, color="#75c995" if dark_mode else "#1d4a36"),
            name="Utility-Based Agent",
            text=utility_df["action"],
            hovertemplate="Step %{x}: Utility %{y:.2f}°C (%{text})<extra></extra>",
        )
    )

    # Simple Reflex Agent Temperature (Muted Teal)
    fig.add_trace(
        go.Scatter(
            x=reflex_df["step"],
            y=reflex_df["actual_temp"],
            mode="lines+markers",
            line=dict(color="#5bb5b5" if dark_mode else "#2e8b8b", width=2.6, dash="solid"),
            marker=dict(size=5, color="#4aa6a6" if dark_mode else "#246f6f", symbol="diamond"),
            name="Simple Reflex Thermostat",
            text=reflex_df["action"],
            hovertemplate="Step %{x}: Reflex %{y:.2f}°C (%{text})<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>1. Temperature Comparison (Synchronized Run)</b>",
        title_font=dict(size=16, color="#e7f0e8" if dark_mode else "#26382d"),
        xaxis_title="Simulation Time Step",
        yaxis_title="Temperature (°C)",
        template="plotly_dark" if dark_mode else "plotly_white",
        height=390,
        margin=dict(l=30, r=20, t=55, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(25, 39, 32, 0.85)" if dark_mode else "rgba(255,255,255,0.75)",
            font=dict(color="#e7f0e8" if dark_mode else "#26382d"),
        ),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(25, 39, 32, 0.6)" if dark_mode else "rgba(245, 247, 240, 0.6)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#304438" if dark_mode else "#dce5d8")
    fig.update_yaxes(showgrid=True, gridcolor="#304438" if dark_mode else "#dce5d8")
    return fig


def create_comparison_energy_chart(
    utility_df: Optional[pd.DataFrame] = None,
    reflex_df: Optional[pd.DataFrame] = None,
    df: Optional[pd.DataFrame] = None,
    dark_mode: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """
    Chart 2: Compare cumulative energy consumption over time for both agents.
    Biophilic Eco-Tech Colors:
    - Utility-Based Agent: Forest Green (#245C43 Light / #8FD0A8 Dark)
    - Simple Reflex Agent: Muted Teal (#2E8B8B Light / #5BB5B5 Dark)
    """
    if utility_df is None and df is not None:
        utility_df = pd.DataFrame({
            "step": df["step"],
            "cumulative_energy": df.get("utility_cumulative_energy", df.get("cumulative_energy", 0.0)),
            "energy_consumed": df.get("utility_energy", df.get("energy_consumed", 0.0)),
        })
        reflex_df = pd.DataFrame({
            "step": df["step"],
            "cumulative_energy": df.get("reflex_cumulative_energy", df.get("cumulative_energy", 0.0)),
            "energy_consumed": df.get("reflex_energy", df.get("energy_consumed", 0.0)),
        })

    fig = go.Figure()
    if utility_df is None or reflex_df is None or utility_df.empty or reflex_df.empty:
        fig.update_layout(title="No energy data recorded yet.", height=320)
        return fig

    fig.add_trace(
        go.Scatter(
            x=utility_df["step"],
            y=utility_df["cumulative_energy"],
            mode="lines+markers",
            line=dict(color="#8fd0a8" if dark_mode else "#245c43", width=3),
            marker=dict(size=5),
            name="Utility-Based Agent",
            customdata=utility_df["energy_consumed"],
            hovertemplate="Step %{x}: Cumulative %{y:.1f} units (Step: %{customdata:.1f})<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=reflex_df["step"],
            y=reflex_df["cumulative_energy"],
            mode="lines+markers",
            line=dict(color="#5bb5b5" if dark_mode else "#2e8b8b", width=2.8),
            marker=dict(size=5, symbol="diamond"),
            name="Simple Reflex Thermostat",
            customdata=reflex_df["energy_consumed"],
            hovertemplate="Step %{x}: Cumulative %{y:.1f} units (Step: %{customdata:.1f})<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>2. Cumulative Energy Consumption Over Time</b>",
        title_font=dict(size=15, color="#e7f0e8" if dark_mode else "#26382d"),
        xaxis_title="Simulation Time Step",
        yaxis_title="Cumulative Energy (units)",
        template="plotly_dark" if dark_mode else "plotly_white",
        height=320,
        margin=dict(l=25, r=20, t=50, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(25, 39, 32, 0.85)" if dark_mode else "rgba(255,255,255,0.7)",
            font=dict(color="#e7f0e8" if dark_mode else "#26382d"),
        ),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(25, 39, 32, 0.6)" if dark_mode else "rgba(245, 247, 240, 0.6)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#304438" if dark_mode else "#dce5d8")
    fig.update_yaxes(showgrid=True, gridcolor="#304438" if dark_mode else "#dce5d8", rangemode="tozero")
    return fig


def create_comparison_error_chart(
    utility_df: Optional[pd.DataFrame] = None,
    reflex_df: Optional[pd.DataFrame] = None,
    df: Optional[pd.DataFrame] = None,
    desired_temp: float = 22.0,
    dark_mode: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """
    Chart 3: Compare absolute temperature error |T_t - T_desired| at every time step.
    Biophilic Eco-Tech Colors:
    - Utility-Based Agent: Forest Green (#245C43 Light / #8FD0A8 Dark)
    - Simple Reflex Agent: Muted Teal (#2E8B8B Light / #5BB5B5 Dark)
    """
    if utility_df is None and df is not None:
        u_temp = df.get("utility_temp", df.get("actual_temp", 0.0))
        r_temp = df.get("reflex_temp", df.get("actual_temp", 0.0))
        utility_df = pd.DataFrame({
            "step": df["step"],
            "abs_error": df.get("utility_abs_error", (u_temp - desired_temp).abs()),
        })
        reflex_df = pd.DataFrame({
            "step": df["step"],
            "abs_error": df.get("reflex_abs_error", (r_temp - desired_temp).abs()),
        })
    elif utility_df is not None and "abs_error" not in utility_df.columns:
        if "actual_temp" in utility_df.columns:
            utility_df = utility_df.copy()
            utility_df["abs_error"] = (utility_df["actual_temp"] - desired_temp).abs()

    if reflex_df is not None and "abs_error" not in reflex_df.columns:
        if "actual_temp" in reflex_df.columns:
            reflex_df = reflex_df.copy()
            reflex_df["abs_error"] = (reflex_df["actual_temp"] - desired_temp).abs()

    fig = go.Figure()
    if utility_df is None or reflex_df is None or utility_df.empty or reflex_df.empty:
        fig.update_layout(title="No temperature error data recorded yet.", height=320)
        return fig

    fig.add_trace(
        go.Scatter(
            x=utility_df["step"],
            y=utility_df["abs_error"],
            mode="lines+markers",
            line=dict(color="#8fd0a8" if dark_mode else "#245c43", width=2.8),
            marker=dict(size=5),
            name="Utility-Based Agent Error",
            hovertemplate="Step %{x}: |Error| = %{y:.2f}°C<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=reflex_df["step"],
            y=reflex_df["abs_error"],
            mode="lines+markers",
            line=dict(color="#5bb5b5" if dark_mode else "#2e8b8b", width=2.6),
            marker=dict(size=5, symbol="diamond"),
            name="Simple Reflex Thermostat Error",
            hovertemplate="Step %{x}: |Error| = %{y:.2f}°C<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>3. Absolute Temperature Error Over Time</b>",
        title_font=dict(size=15, color="#e7f0e8" if dark_mode else "#26382d"),
        xaxis_title="Simulation Time Step",
        yaxis_title="Absolute Error (°C)",
        template="plotly_dark" if dark_mode else "plotly_white",
        height=320,
        margin=dict(l=25, r=20, t=50, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(25, 39, 32, 0.85)" if dark_mode else "rgba(255,255,255,0.7)",
            font=dict(color="#e7f0e8" if dark_mode else "#26382d"),
        ),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(25, 39, 32, 0.6)" if dark_mode else "rgba(245, 247, 240, 0.6)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#304438" if dark_mode else "#dce5d8")
    fig.update_yaxes(showgrid=True, gridcolor="#304438" if dark_mode else "#dce5d8", rangemode="tozero")
    return fig


def create_comparison_action_distribution_chart(
    utility_df: Optional[pd.DataFrame] = None,
    reflex_df: Optional[pd.DataFrame] = None,
    df: Optional[pd.DataFrame] = None,
    dark_mode: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """
    Chart 4: Compare HEAT, COOL, and OFF activation frequencies for both agents.
    Biophilic Eco-Tech Colors:
    - Utility-Based Agent: Forest Green (#245C43 Light / #8FD0A8 Dark)
    - Simple Reflex Agent: Muted Teal (#2E8B8B Light / #5BB5B5 Dark)
    """
    if utility_df is None and df is not None:
        utility_df = pd.DataFrame({
            "step": df["step"],
            "action": df.get("utility_action", df.get("action", "")),
        })
        reflex_df = pd.DataFrame({
            "step": df["step"],
            "action": df.get("reflex_action", df.get("action", "")),
        })

    u_active = utility_df[utility_df["step"] > 0] if utility_df is not None and not utility_df.empty else pd.DataFrame()
    r_active = reflex_df[reflex_df["step"] > 0] if reflex_df is not None and not reflex_df.empty else pd.DataFrame()

    actions = [ACTION_HEAT, ACTION_COOL, ACTION_OFF]
    labels = ["🔥 HEAT", "❄️ COOL", "⏸️ OFF"]

    u_counts = [int((u_active["action"] == a).sum()) if not u_active.empty else 0 for a in actions]
    r_counts = [int((r_active["action"] == a).sum()) if not r_active.empty else 0 for a in actions]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Utility-Based Agent",
            x=labels,
            y=u_counts,
            marker_color="#8fd0a8" if dark_mode else "#245c43",
            text=[str(c) for c in u_counts],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Simple Reflex Thermostat",
            x=labels,
            y=r_counts,
            marker_color="#5bb5b5" if dark_mode else "#2e8b8b",
            text=[str(c) for c in r_counts],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="<b>4. Action Distribution (HEAT / COOL / OFF)</b>",
        title_font=dict(size=15, color="#e7f0e8" if dark_mode else "#26382d"),
        barmode="group",
        yaxis_title="Number of Steps",
        template="plotly_dark" if dark_mode else "plotly_white",
        height=310,
        margin=dict(l=25, r=20, t=50, b=25),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(25, 39, 32, 0.85)" if dark_mode else "rgba(255,255,255,0.7)",
            font=dict(color="#e7f0e8" if dark_mode else "#26382d"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(25, 39, 32, 0.6)" if dark_mode else "rgba(245, 247, 240, 0.6)",
    )
    fig.update_yaxes(showgrid=True, gridcolor="#304438" if dark_mode else "#dce5d8", rangemode="tozero")
    return fig


def create_comparison_classification_chart(
    util_metrics: Any = None,
    refl_metrics: Any = None,
    dark_mode: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """
    Chart 5: Compare percentage of time each agent spends in Cold, Comfortable,
    and Hot environment classifications.
    Biophilic Eco-Tech Colors:
    - Utility-Based Agent: Forest Green (#245C43 Light / #8FD0A8 Dark)
    - Simple Reflex Agent: Muted Teal (#2E8B8B Light / #5BB5B5 Dark)
    """
    categories = ["❄️ Cold", "🌿 Comfortable", "🔥 Hot"]
    u_pcts = [
        getattr(util_metrics, "cold_percentage", 0.0) if util_metrics else 0.0,
        getattr(util_metrics, "comfortable_percentage", 0.0) if util_metrics else 0.0,
        getattr(util_metrics, "hot_percentage", 0.0) if util_metrics else 0.0,
    ]
    r_pcts = [
        getattr(refl_metrics, "cold_percentage", 0.0) if refl_metrics else 0.0,
        getattr(refl_metrics, "comfortable_percentage", 0.0) if refl_metrics else 0.0,
        getattr(refl_metrics, "hot_percentage", 0.0) if refl_metrics else 0.0,
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Utility-Based Agent",
            x=categories,
            y=u_pcts,
            marker_color="#8fd0a8" if dark_mode else "#245c43",
            text=[f"{v:.1f}%" for v in u_pcts],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Simple Reflex Thermostat",
            x=categories,
            y=r_pcts,
            marker_color="#5bb5b5" if dark_mode else "#2e8b8b",
            text=[f"{v:.1f}%" for v in r_pcts],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="<b>5. Environment Classification Time Share (%)</b>",
        title_font=dict(size=15, color="#e7f0e8" if dark_mode else "#26382d"),
        barmode="group",
        yaxis_title="Percentage of Steps (%)",
        yaxis_range=[0, 108],
        template="plotly_dark" if dark_mode else "plotly_white",
        height=310,
        margin=dict(l=25, r=20, t=50, b=25),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(25, 39, 32, 0.85)" if dark_mode else "rgba(255,255,255,0.7)",
            font=dict(color="#e7f0e8" if dark_mode else "#26382d"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(25, 39, 32, 0.6)" if dark_mode else "rgba(245, 247, 240, 0.6)",
    )
    fig.update_yaxes(showgrid=True, gridcolor="#304438" if dark_mode else "#dce5d8")
    return fig
