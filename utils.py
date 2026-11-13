"""
Helper functions and visualization utilities for the Environment Classifier
and Demonstrator application (Environment, Utility Agent, and Reflex Thermostat).
Biophilic Eco-Tech Design System: Contrast, Typography & Chart Layout.
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


def _clean_html(html: str) -> str:
    """
    Ensure every line of generated HTML is stripped of leading indentation
    and remove all blank lines so CommonMark never prematurely terminates
    HTML block parsing.
    """
    lines = [line.strip() for line in html.splitlines()]
    return "\n".join(line for line in lines if line)


def _get_theme_palette(dark_mode: bool) -> Dict[str, str]:
    """
    Biophilic Eco-Tech Theme Palette source of truth.
    Provides verified high-contrast colors and readable typography tokens.
    """
    if dark_mode:
        return {
            "app_bg": "#101B17",
            "card_bg": "#192720",
            "sub_bg": "#22352A",
            "text_primary": "#E7F0E8",
            "text_secondary": "#B4C4B5",
            "text_muted": "#8DA08F",
            "primary_green": "#75B58A",
            "forest_green": "#75B58A",
            "sage_green": "#9DBA91",
            "border_color": "#304438",
            "success": "#75B58A",
            "warning": "#E8B65B",
            "danger": "#E58B80",
            "util_agent": "#75B58A",
            "reflex_agent": "#5BB5B5",
            "target": "#9DBA91",
            "outside": "#B4C4B5",
            "heating": "#E8B65B",
            "cooling": "#7AB8DA",
            "plot_bg": "rgba(25, 39, 32, 0.65)",
            "grid_color": "#304438",
            "legend_bg": "rgba(25, 39, 32, 0.92)",
            "comfort_fill": "rgba(117, 181, 138, 0.16)",
            "deadband_fill": "rgba(157, 186, 145, 0.16)",
            "status_cold": "#6DA4C7",
            "status_cold_bg": "#162732",
            "status_cold_border": "#294356",
            "status_comf": "#75B58A",
            "status_comf_bg": "#1B2D22",
            "status_comf_border": "#2D4837",
            "status_hot": "#E0A05A",
            "status_hot_bg": "#2E2216",
            "status_hot_border": "#523A22",
        }
    else:
        return {
            "app_bg": "#F5F7F0",
            "card_bg": "#FFFFFF",
            "sub_bg": "#EAF2EB",
            "text_primary": "#26382D",
            "text_secondary": "#5F7162",
            "text_muted": "#718174",
            "primary_green": "#245C43",
            "forest_green": "#245C43",
            "sage_green": "#829B78",
            "border_color": "#DCE5D8",
            "success": "#245C43",
            "warning": "#D99A35",
            "danger": "#C65B52",
            "util_agent": "#245C43",
            "reflex_agent": "#4D9285",
            "target": "#829B78",
            "outside": "#5F7162",
            "heating": "#D99A35",
            "cooling": "#5D9FC2",
            "plot_bg": "rgba(245, 247, 240, 0.65)",
            "grid_color": "#DCE5D8",
            "legend_bg": "rgba(255, 255, 255, 0.92)",
            "comfort_fill": "rgba(36, 92, 67, 0.12)",
            "deadband_fill": "rgba(130, 155, 120, 0.15)",
            "status_cold": "#2A6282",
            "status_cold_bg": "#EBF3F7",
            "status_cold_border": "#BDD8E6",
            "status_comf": "#245C43",
            "status_comf_bg": "#EAF4EC",
            "status_comf_border": "#C6DEC9",
            "status_hot": "#965C1E",
            "status_hot_bg": "#FDF4E7",
            "status_hot_border": "#F0D6B3",
        }


def _apply_standard_chart_layout(
    fig: go.Figure,
    title_text: str,
    dark_mode: bool,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    height: int = 420,
    show_legend: bool = True,
    hovermode: Optional[str] = "x unified",
    margin_top: int = 100,
    margin_bottom: int = 70,
    margin_left: int = 75,
    margin_right: int = 35,
) -> go.Figure:
    """
    Apply a consistent, professional Plotly layout that prevents overlapping
    titles and legends, ensures readable axes, and respects the Biophilic theme.
    """
    p = _get_theme_palette(dark_mode)
    font_family = "'DM Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif"

    layout_args: Dict[str, Any] = dict(
        autosize=True,
        title=dict(
            text=title_text,
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            font=dict(
                size=16,
                color=p["text_primary"],
                family=font_family,
            ),
        ),
        template="plotly_dark" if dark_mode else "plotly_white",
        height=height,
        margin=dict(l=margin_left, r=margin_right, t=margin_top, b=margin_bottom),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=p["plot_bg"],
        font=dict(
            family=font_family,
            color=p["text_secondary"],
            size=12,
        ),
    )

    if show_legend:
        layout_args["legend"] = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.02,
            bgcolor=p["legend_bg"],
            bordercolor=p["border_color"],
            borderwidth=1,
            font=dict(
                size=12,
                color=p["text_secondary"],
                family=font_family,
            ),
        )

    if hovermode:
        layout_args["hovermode"] = hovermode

    if xaxis_title:
        layout_args["xaxis_title"] = xaxis_title
    if yaxis_title:
        layout_args["yaxis_title"] = yaxis_title

    fig.update_layout(**layout_args)

    fig.update_xaxes(
        showgrid=True,
        gridcolor=p["grid_color"],
        title_font=dict(size=13, color=p["text_primary"], family=font_family),
        tickfont=dict(size=12, color=p["text_secondary"], family=font_family),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=p["grid_color"],
        title_font=dict(size=13, color=p["text_primary"], family=font_family),
        tickfont=dict(size=12, color=p["text_secondary"], family=font_family),
    )

    return fig


# ---------------------------------------------------------
# HTML Card Generators
# ---------------------------------------------------------
def get_status_card_html(classification: str, current_temp: float, desired_temp: float, dark_mode: bool = False) -> str:
    """
    Generate an HTML status card representing the temperature classification
    using the Biophilic Eco-Tech design palette with high-contrast text and Lucide outline icons.
    """
    p = _get_theme_palette(dark_mode)
    delta = current_temp - desired_temp
    delta_str = f"{'+' if delta > 0 else ''}{delta:.1f}°C from target ({desired_temp:.1f}°C)"

    if classification == CLASS_COLD:
        accent_color = p["status_cold"]
        badge_bg = p["status_cold_bg"]
        badge_border = p["status_cold_border"]
        desc = "Temperature is below the comfortable range. Heating actuation recommended."
        icon_svg = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/><line x1="4" y1="9" x2="8" y2="9"/><line x1="6" y1="7" x2="6" y2="11"/><line x1="2" y1="14" x2="6" y2="14"/><line x1="4" y1="12" x2="4" y2="16"/></svg>"""
    elif classification == CLASS_HOT:
        accent_color = p["status_hot"]
        badge_bg = p["status_hot_bg"]
        badge_border = p["status_hot_border"]
        desc = "Temperature is above the comfortable range. Cooling actuation recommended."
        icon_svg = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9a4 4 0 0 0-2 7.5v.5a2 2 0 0 0 4 0v-.5A4 4 0 0 0 12 9z"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>"""
    else:
        accent_color = p["status_comf"]
        badge_bg = p["status_comf_bg"]
        badge_border = p["status_comf_border"]
        desc = "Temperature is within the optimal comfort zone. Energy conservation active."
        icon_svg = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6"/></svg>"""

    html = f"""<div style="background-color: {p['card_bg']}; border: 1px solid {p['border_color']}; border-left: 4px solid {accent_color}; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);">
<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
<div style="display: flex; align-items: center; gap: 14px;">
<div style="display: flex; align-items: center; justify-content: center; width: 44px; height: 44px; border-radius: 10px; background-color: {badge_bg}; border: 1px solid {badge_border}; color: {accent_color};">
{icon_svg}
</div>
<div>
<div style="font-size: 11.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; color: {accent_color};">Environment Classification</div>
<div style="font-size: 22px; font-weight: 700; color: {p['text_primary']}; line-height: 1.2;">{classification.upper()}</div>
</div>
</div>
<div style="text-align: right;">
<div style="font-size: 28px; font-weight: 700; color: {accent_color}; line-height: 1.15; font-feature-settings: 'tnum' 1;">{current_temp:.1f}°C</div>
<div style="font-size: 13px; font-weight: 400; color: {p['text_secondary']}; margin-top: 2px;">{delta_str}</div>
</div>
</div>
<div style="margin-top: 12px; font-size: 13.5px; color: {p['text_secondary']}; border-top: 1px solid {p['border_color']}; padding-top: 10px; line-height: 1.45;">
{desc}
</div>
</div>"""
    return _clean_html(html)


def get_overview_metric_card_html(
    label: str,
    value: str,
    subtext: str,
    status_type: Optional[str] = None,
    dark_mode: bool = False,
) -> str:
    """
    Generate an uncluttered, premium Biophilic metric card without placeholder
    boxes or redundant icons. Values are prominent with subtle inline supporting text.
    """
    p = _get_theme_palette(dark_mode)
    val_color = p["text_primary"]

    if status_type == "cold":
        val_color = p["status_cold"]
    elif status_type == "hot":
        val_color = p["status_hot"]
    elif status_type == "comfortable":
        val_color = p["status_comf"]

    html = f"""<div style="background-color: {p['card_bg']}; border: 1px solid {p['border_color']}; border-radius: 10px; padding: 14px 16px; height: 100%; box-sizing: border-box; display: flex; flex-direction: column; justify-content: space-between;">
<div style="font-size: 13px; font-weight: 500; color: {p['text_secondary']}; margin-bottom: 4px;">{label}</div>
<div style="font-size: 28px; font-weight: 700; color: {val_color}; line-height: 1.15; margin-bottom: 6px; font-feature-settings: 'tnum' 1;">{value}</div>
<div style="font-size: 13px; font-weight: 400; color: {p['text_secondary']};">{subtext}</div>
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
    in Biophilic Eco-Tech palette with high-contrast text.
    """
    p = _get_theme_palette(dark_mode)
    meta = ACTION_METADATA.get(selected_action, ACTION_METADATA[ACTION_OFF])
    action_accent = meta['color']

    html = f"""<div style="background: {p['card_bg']}; border: 1.5px solid {p['border_color']}; border-left: 5px solid {action_accent}; border-radius: 10px; padding: 14px 18px; margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
<div>
<div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: {p['text_secondary']};">Utility Agent Decision</div>
<div style="font-size: 20px; font-weight: 800; color: {p['text_primary']};">Selected Action: <span style="color: {action_accent};">{meta['label']}</span></div>
</div>
<div style="text-align: right;">
<div style="font-size: 11px; color: {p['text_secondary']}; font-weight: 600; text-transform: uppercase;">Max Utility Score</div>
<div style="font-size: 22px; font-weight: 800; color: {p['text_primary']};">{utility_score:.2f}</div>
</div>
</div>
<div style="display: flex; gap: 12px; background: {p['sub_bg']}; padding: 10px 14px; border-radius: 8px; border: 1px solid {p['border_color']}; margin-bottom: 12px;">
<div style="flex: 1;">
<div style="font-size: 11px; color: {p['text_secondary']}; font-weight: 600;">Current Temp</div>
<div style="font-size: 16px; font-weight: 700; color: {p['text_primary']};">{current_temp:.2f}°C</div>
</div>
<div style="flex: 1;">
<div style="font-size: 11px; color: {p['text_secondary']}; font-weight: 600;">Desired Target</div>
<div style="font-size: 16px; font-weight: 700; color: {p['success']};">{desired_temp:.1f}°C</div>
</div>
<div style="flex: 1;">
<div style="font-size: 11px; color: {p['text_secondary']}; font-weight: 600;">Predicted Outcome</div>
<div style="font-size: 16px; font-weight: 700; color: {action_accent};">{predicted_temp:.2f}°C</div>
</div>
</div>
<div style="font-size: 13.5px; color: {p['text_primary']}; line-height: 1.45; padding: 8px 12px; background: {p['sub_bg']}; border-radius: 6px; border: 1px solid {p['border_color']};">
<b>💡 Decision Reason:</b> {reason}
</div>
</div>"""
    return _clean_html(html)


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
    p = _get_theme_palette(dark_mode)

    rules = [
        {
            "action": ACTION_HEAT,
            "condition": f"&lt; {lower_threshold:.1f}°C",
            "arrow_label": "🔥 HEAT",
            "desc": "Room is below lower threshold — turn on heating.",
            "accent": p["danger"],
            "bg_active": p["sub_bg"],
            "border_active": p["danger"],
        },
        {
            "action": ACTION_OFF,
            "condition": f"{lower_threshold:.1f}°C – {upper_threshold:.1f}°C",
            "arrow_label": "⏸️ OFF",
            "desc": "Room is between thresholds — turn off both systems.",
            "accent": p["success"],
            "bg_active": p["sub_bg"],
            "border_active": p["success"],
        },
        {
            "action": ACTION_COOL,
            "condition": f"&gt; {upper_threshold:.1f}°C",
            "arrow_label": "❄️ COOL",
            "desc": "Room is above upper threshold — turn on cooling.",
            "accent": p["reflex_agent"],
            "bg_active": p["sub_bg"],
            "border_active": p["reflex_agent"],
        },
    ]

    cards_html = []
    for rule in rules:
        is_active = rule["action"] == selected_action
        bg = rule["bg_active"] if is_active else p["card_bg"]
        border = f"2px solid {rule['border_active']}" if is_active else f"1.5px solid {p['border_color']}"
        opacity = "1.0" if is_active else "0.85"
        shadow = "0 3px 8px rgba(0, 0, 0, 0.12)" if is_active else "none"
        badge_bg = rule["accent"] if is_active else p["sage_green"]
        badge_text = "ACTIVE RULE" if is_active else "INACTIVE"

        cards_html.append(
            f"""<div style="flex: 1; min-width: 210px; background-color: {bg}; border: {border}; border-radius: 10px; padding: 14px 16px; opacity: {opacity}; box-shadow: {shadow};">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
<span style="font-size: 11px; font-weight: 700; letter-spacing: 0.05em; color: white; background-color: {badge_bg}; padding: 2px 8px; border-radius: 999px;">{badge_text}</span>
<span style="font-size: 12px; font-weight: 600; color: {p['text_secondary']};">Current: {current_temp:.2f}°C</span>
</div>
<div style="font-size: 18px; font-weight: 800; color: {p['text_primary']}; margin-bottom: 4px;">
{rule['condition']} <span style="color: {p['text_secondary']}; margin: 0 4px;">→</span> <span style="color: {rule['accent']};">{rule['arrow_label']}</span>
</div>
<div style="font-size: 12.5px; color: {p['text_secondary']}; line-height: 1.4;">{rule['desc']}</div>
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
    in Biophilic Eco-Tech palette with high-contrast text.
    """
    p = _get_theme_palette(dark_mode)
    meta = ACTION_METADATA.get(selected_action, ACTION_METADATA[ACTION_OFF])
    class_info = STATUS_COLORS.get(classification, STATUS_COLORS[CLASS_COMFORTABLE])

    last_step_note = ""
    if last_step_num > 0 and last_step_action:
        last_meta = ACTION_METADATA.get(last_step_action, ACTION_METADATA[ACTION_OFF])
        last_step_note = (
            f'<div style="margin-top: 8px; font-size: 12.5px; color: {p["text_secondary"]};">'
            f'<b>Last Executed Step (Step {last_step_num}):</b> '
            f'<span style="color: {last_meta["color"]}; font-weight: 700;">{last_meta["label"]}</span> '
            f'(Step energy: {step_energy:.1f} units)'
            f"</div>"
        )

    html = f"""<div style="background: {p['card_bg']}; border: 1.5px solid {p['border_color']}; border-left: 5px solid {meta['color']}; border-radius: 10px; padding: 14px 18px; margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
<div>
<div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: {p['text_secondary']};">Simple Reflex Thermostat Decision (Current Percept)</div>
<div style="font-size: 20px; font-weight: 800; color: {p['text_primary']};">Selected Action: <span style="color: {meta['color']};">{meta['label']}</span></div>
</div>
<div style="text-align: right;">
<div style="font-size: 11px; color: {p['text_secondary']}; font-weight: 600; text-transform: uppercase;">Triggered Rule</div>
<div style="font-size: 16px; font-weight: 800; color: {p['text_primary']};">{rule_triggered}</div>
</div>
</div>
<div style="display: flex; gap: 12px; flex-wrap: wrap; background: {p['sub_bg']}; padding: 10px 14px; border-radius: 8px; border: 1px solid {p['border_color']}; margin-bottom: 12px;">
<div style="flex: 1; min-width: 110px;">
<div style="font-size: 11px; color: {p['text_secondary']}; font-weight: 600;">Room Temp</div>
<div style="font-size: 16px; font-weight: 700; color: {p['text_primary']};">{current_temp:.2f}°C</div>
</div>
<div style="flex: 1; min-width: 110px;">
<div style="font-size: 11px; color: {p['text_secondary']}; font-weight: 600;">Lower Threshold</div>
<div style="font-size: 16px; font-weight: 700; color: {p['danger']};">{lower_threshold:.1f}°C</div>
</div>
<div style="flex: 1; min-width: 110px;">
<div style="font-size: 11px; color: {p['text_secondary']}; font-weight: 600;">Upper Threshold</div>
<div style="font-size: 16px; font-weight: 700; color: {p['reflex_agent']};">{upper_threshold:.1f}°C</div>
</div>
<div style="flex: 1; min-width: 110px;">
<div style="font-size: 11px; color: {p['text_secondary']}; font-weight: 600;">Classification</div>
<div style="font-size: 16px; font-weight: 700; color: {class_info['color']};">{class_info['icon']} {classification}</div>
</div>
<div style="flex: 1; min-width: 110px;">
<div style="font-size: 11px; color: {p['text_secondary']}; font-weight: 600;">Energy Used</div>
<div style="font-size: 16px; font-weight: 700; color: {p['text_primary']};">{total_energy:.1f} units</div>
</div>
</div>
<div style="font-size: 13.5px; color: {p['text_primary']}; line-height: 1.45; padding: 8px 12px; background: {p['sub_bg']}; border-radius: 6px; border: 1px solid {p['border_color']};">
<b>⚡ Condition-Action Rule Match:</b> {reason}
{last_step_note}
</div>
</div>"""
    return _clean_html(html)


def get_comparison_summary_html(summary: Dict[str, Any], dark_mode: bool = False) -> str:
    """
    Generate an overall comparison summary banner driven strictly by actual
    simulation metric wins (never hardcoded) in Biophilic Eco-Tech palette.
    """
    p = _get_theme_palette(dark_mode)
    winner = summary.get("overall_winner", "Tie")
    u_wins = summary.get("utility_wins", 0)
    r_wins = summary.get("reflex_wins", 0)
    ties = summary.get("ties", 0)
    headline = summary.get("headline", "")

    if winner == "Utility-Based Agent":
        accent = p["util_agent"]
        bg = p["sub_bg"]
        border = p["primary_green"]
        badge = "🌱 UTILITY-BASED AGENT LEADS"
    elif winner == "Simple Reflex Agent":
        accent = p["reflex_agent"]
        bg = p["sub_bg"]
        border = p["reflex_agent"]
        badge = "🔁 SIMPLE REFLEX THERMOSTAT LEADS"
    else:
        accent = p["sage_green"]
        bg = p["card_bg"]
        border = p["border_color"]
        badge = "⚖️ BALANCED / EQUAL PERFORMANCE"

    html = f"""<div style="background-color: {p['card_bg']}; border: 2px solid {border}; border-radius: 12px; padding: 16px 20px; margin-bottom: 18px; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
<div>
<div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: {accent};">Overall Comparison Summary (Data-Driven)</div>
<div style="font-size: 22px; font-weight: 800; color: {p['text_primary']}; margin-top: 2px;">{badge}</div>
<div style="font-size: 14px; color: {p['text_secondary']}; margin-top: 4px;">{headline}</div>
</div>
<div style="display: flex; gap: 12px; flex-wrap: wrap;">
<div style="background: {p['sub_bg']}; border: 1px solid {p['border_color']}; border-radius: 8px; padding: 8px 14px; text-align: center;">
<div style="font-size: 11px; font-weight: 700; color: {p['util_agent']};">UTILITY WINS</div>
<div style="font-size: 20px; font-weight: 800; color: {p['text_primary']};">{u_wins}</div>
</div>
<div style="background: {p['sub_bg']}; border: 1px solid {p['border_color']}; border-radius: 8px; padding: 8px 14px; text-align: center;">
<div style="font-size: 11px; font-weight: 700; color: {p['reflex_agent']};">REFLEX WINS</div>
<div style="font-size: 20px; font-weight: 800; color: {p['text_primary']};">{r_wins}</div>
</div>
<div style="background: {p['sub_bg']}; border: 1px solid {p['border_color']}; border-radius: 8px; padding: 8px 14px; text-align: center;">
<div style="font-size: 11px; font-weight: 700; color: {p['text_muted']};">TIES</div>
<div style="font-size: 20px; font-weight: 800; color: {p['text_primary']};">{ties}</div>
</div>
</div>
</div>
</div>"""
    return _clean_html(html)


# ---------------------------------------------------------
# Plotly Chart Generators
# ---------------------------------------------------------
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
    using Biophilic Eco-Tech palette with compact vertical footprint.
    """
    p = _get_theme_palette(dark_mode)
    lower_bound = max(min_gauge, desired_temp - comfort_tolerance)
    upper_bound = min(max_gauge, desired_temp + comfort_tolerance)

    if current_temp < lower_bound:
        bar_color = p["cooling"]
    elif current_temp > upper_bound:
        bar_color = p["heating"]
    else:
        bar_color = p["primary_green"]

    step_cold = "rgba(109, 164, 199, 0.16)" if dark_mode else "rgba(42, 98, 130, 0.10)"
    step_comfort = "rgba(117, 181, 138, 0.18)" if dark_mode else "rgba(36, 92, 67, 0.12)"
    step_hot = "rgba(224, 160, 90, 0.16)" if dark_mode else "rgba(150, 92, 30, 0.10)"

    font_family = "'DM Sans', 'Inter', -apple-system, sans-serif"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=current_temp,
            number={"suffix": "°C", "font": {"size": 30, "color": p["text_primary"], "family": font_family}},
            delta={
                "reference": desired_temp,
                "increasing": {"color": p["heating"]},
                "decreasing": {"color": p["cooling"]},
                "suffix": "° vs Target",
                "font": {"size": 12, "family": font_family},
            },
            title={"text": "<b>Current Room Temperature</b>", "font": {"size": 14, "color": p["text_primary"], "family": font_family}},
            gauge={
                "axis": {
                    "range": [min_gauge, max_gauge],
                    "tickwidth": 1,
                    "tickcolor": p["text_secondary"],
                    "tickfont": {"size": 11, "color": p["text_secondary"], "family": font_family},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": p["card_bg"],
                "borderwidth": 1,
                "bordercolor": p["border_color"],
                "steps": [
                    {"range": [min_gauge, lower_bound], "color": step_cold},
                    {"range": [lower_bound, upper_bound], "color": step_comfort},
                    {"range": [upper_bound, max_gauge], "color": step_hot},
                ],
                "threshold": {
                    "line": {"color": p["target"], "width": 3.0},
                    "thickness": 0.75,
                    "value": desired_temp,
                },
            },
        )
    )

    fig.update_layout(
        autosize=True,
        height=270,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=font_family, color=p["text_secondary"]),
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
    in Biophilic palette with generous width usage and high-contrast labels.
    """
    p = _get_theme_palette(dark_mode)
    categories = ["Current Indoor", "Target", "Outside Ambient"]
    temperatures = [room_temp, desired_temp, outside_temp]
    colors = [p["util_agent"], p["target"], p["sage_green"]]

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=temperatures,
            marker_color=colors,
            marker_line=dict(color=p["border_color"], width=1),
            text=[f"{t:.1f}°C" for t in temperatures],
            textposition="auto",
            textfont=dict(size=13, color="white", weight="bold"),
            width=0.55,
        )
    )

    _apply_standard_chart_layout(
        fig=fig,
        title_text="<b>Temperature Comparison</b>",
        dark_mode=dark_mode,
        yaxis_title="Temperature (°C)",
        height=270,
        show_legend=False,
        hovermode=None,
        margin_top=65,
        margin_bottom=45,
        margin_left=65,
        margin_right=25,
    )
    return fig


def create_utility_comparison_chart(
    action_evaluations: Dict[str, Any],
    selected_action: str,
    dark_mode: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """
    Generate a bar chart comparing the total utility scores for HEAT, COOL, and OFF.
    The selected action is highlighted with Forest Green in the Biophilic theme.
    """
    p = _get_theme_palette(dark_mode)
    actions = [ACTION_HEAT, ACTION_COOL, ACTION_OFF]
    labels = ["HEAT (+Power)", "COOL (-Power)", "OFF (Drift)"]
    utilities = [action_evaluations[a].utility if a in action_evaluations else 0.0 for a in actions]

    winner_color = p["primary_green"]
    winner_border = p["sage_green"]
    other_color = p["sub_bg"]
    other_border = p["border_color"]

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
            textfont=dict(size=14, weight="bold", color=p["text_primary"]),
            width=0.45,
        )
    )

    _apply_standard_chart_layout(
        fig=fig,
        title_text="<b>Action Utility Comparison (Highest Selected)</b>",
        dark_mode=dark_mode,
        yaxis_title="Net Utility Score",
        height=350,
        show_legend=False,
        hovermode=None,
        margin_top=80,
        margin_bottom=55,
        margin_left=75,
        margin_right=35,
    )
    return fig


def create_utility_factors_breakdown_chart(
    action_evaluations: Dict[str, Any],
    dark_mode: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """
    Generate a grouped/stacked bar chart illustrating the 4 penalty factors:
    Comfort, Energy, Instability, and Overshoot for HEAT, COOL, and OFF.
    """
    p = _get_theme_palette(dark_mode)
    actions = [ACTION_HEAT, ACTION_COOL, ACTION_OFF]
    labels = ["HEAT", "COOL", "OFF"]

    comfort_penalties = [action_evaluations[a].weighted_comfort for a in actions]
    energy_costs = [action_evaluations[a].weighted_energy for a in actions]
    instability_penalties = [action_evaluations[a].weighted_stability for a in actions]
    overshoot_penalties = [action_evaluations[a].weighted_overshoot for a in actions]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Comfort Penalty", x=labels, y=comfort_penalties, marker_color=p["danger"]))
    fig.add_trace(go.Bar(name="Energy Cost", x=labels, y=energy_costs, marker_color=p["warning"]))
    fig.add_trace(go.Bar(name="Instability Penalty", x=labels, y=instability_penalties, marker_color=p["reflex_agent"]))
    fig.add_trace(go.Bar(name="Overshoot Penalty", x=labels, y=overshoot_penalties, marker_color=p["sage_green"]))

    _apply_standard_chart_layout(
        fig=fig,
        title_text="<b>Weighted Penalty Factors Decomposition (Lower is Better)</b>",
        dark_mode=dark_mode,
        yaxis_title="Weighted Cost",
        height=380,
        show_legend=True,
        hovermode="x unified",
        margin_top=105,
        margin_bottom=65,
        margin_left=75,
        margin_right=35,
    )
    fig.update_layout(barmode="stack")
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
    p = _get_theme_palette(dark_mode)
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
            fillcolor=p["comfort_fill"],
            line=dict(color="rgba(0, 0, 0, 0)"),
            hoverinfo="skip",
            showlegend=True,
            name=f"Comfort Zone (±{comfort_tolerance:.1f}°C)",
        )
    )

    # 2. Desired Target Temperature (Center Line)
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=[desired_temp] * len(steps),
            mode="lines",
            line=dict(color=p["target"], width=2.0, dash="dash"),
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
            line=dict(color=p["outside"], width=1.8, dash="dot"),
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
                line=dict(color=p["sage_green"], width=1.8, dash="dashdot"),
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
            line=dict(color=p["util_agent"], width=3.0),
            marker=dict(size=5.5, color=p["primary_green"]),
            name="Actual Room Temp",
            hovertemplate="Step %{x}: Room %{y:.2f}°C (%{text})<extra></extra>",
            text=df["action"] if "action" in df.columns else None,
        )
    )

    _apply_standard_chart_layout(
        fig=fig,
        title_text="<b>Room Temperature Trajectory & Agent Performance</b>",
        dark_mode=dark_mode,
        xaxis_title="Simulation Time Step",
        yaxis_title="Temperature (°C)",
        height=440,
        show_legend=True,
        hovermode="x unified",
        margin_top=110,
        margin_bottom=70,
        margin_left=75,
        margin_right=35,
    )
    return fig


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
    p = _get_theme_palette(dark_mode)
    fig = go.Figure()

    if df.empty:
        fig.update_layout(title="No simulation steps executed yet.", height=380)
        return fig

    steps = df["step"]
    temp_col = "actual_temp" if "actual_temp" in df.columns else "room_temp"
    action_colors = {
        ACTION_HEAT: p["danger"],
        ACTION_COOL: p["reflex_agent"],
        ACTION_OFF: p["success"],
    }
    marker_colors = [
        action_colors.get(a, p["text_muted"])
        for a in (df["action"] if "action" in df.columns else [ACTION_OFF] * len(df))
    ]

    # 1. Shaded OFF Deadband Zone between lower and upper thresholds
    fig.add_trace(
        go.Scatter(
            x=list(steps) + list(steps[::-1]),
            y=[upper_threshold] * len(steps) + [lower_threshold] * len(steps),
            fill="toself",
            fillcolor=p["deadband_fill"],
            line=dict(color="rgba(0, 0, 0, 0)"),
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
            line=dict(color=p["reflex_agent"], width=2.0, dash="dash"),
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
            line=dict(color=p["danger"], width=2.0, dash="dash"),
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
            line=dict(color=p["target"], width=1.5, dash="dot"),
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
            line=dict(color=p["outside"], width=1.6, dash="dot"),
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
            line=dict(color=p["text_primary"], width=2.8),
            marker=dict(size=7, color=marker_colors, line=dict(width=1, color="white")),
            name="Room Temperature",
            text=df["action"] if "action" in df.columns else None,
            hovertemplate="Step %{x}: Room %{y:.2f}°C (Action: %{text})<extra></extra>",
        )
    )

    _apply_standard_chart_layout(
        fig=fig,
        title_text="<b>Simple Reflex Thermostat Temperature Trajectory</b>",
        dark_mode=dark_mode,
        xaxis_title="Simulation Time Step",
        yaxis_title="Temperature (°C)",
        height=440,
        show_legend=True,
        hovermode="x unified",
        margin_top=110,
        margin_bottom=70,
        margin_left=75,
        margin_right=35,
    )
    return fig


def create_reflex_action_summary_chart(df: pd.DataFrame, dark_mode: bool = False, **kwargs: Any) -> go.Figure:
    """
    Generate a bar chart showing the count of HEAT, COOL, and OFF actions
    taken by the Simple Reflex Thermostat during the simulation.
    """
    p = _get_theme_palette(dark_mode)
    active_df = df[df["step"] > 0] if "step" in df.columns and len(df) > 1 else pd.DataFrame()
    actions = [ACTION_HEAT, ACTION_OFF, ACTION_COOL]
    labels = ["HEAT", "OFF", "COOL"]
    colors = [p["heating"], p["primary_green"], p["cooling"]]

    if active_df.empty or "action" not in active_df.columns:
        counts = [0, 0, 0]
    else:
        counts = [int((active_df["action"] == a).sum()) for a in actions]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=counts,
            marker_color=colors,
            marker_line=dict(color=p["border_color"], width=1),
            text=[f"{c} steps" for c in counts],
            textposition="auto",
            textfont=dict(size=13, color="white", weight="bold"),
            width=0.52,
        )
    )

    _apply_standard_chart_layout(
        fig=fig,
        title_text="<b>Reflex Action Distribution</b>",
        dark_mode=dark_mode,
        yaxis_title="Number of Steps",
        height=320,
        show_legend=False,
        hovermode=None,
        margin_top=80,
        margin_bottom=55,
        margin_left=75,
        margin_right=35,
    )
    fig.update_yaxes(dtick=1 if max(counts, default=0) <= 10 else None)
    return fig


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
    - Utility-Based Agent: Forest Green (#245C43 Light / #75B58A Dark)
    - Simple Reflex Agent: Muted Teal (#4D9285 Light / #5BB5B5 Dark)
    - Desired Target: Dashed Natural Green (#39845A Light / #75C995 Dark)
    """
    p = _get_theme_palette(dark_mode)

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

    # 1. Shaded Comfort Range
    fig.add_trace(
        go.Scatter(
            x=list(steps) + list(steps[::-1]),
            y=[upper_bound] * len(steps) + [lower_bound] * len(steps),
            fill="toself",
            fillcolor=p["comfort_fill"],
            line=dict(color="rgba(0, 0, 0, 0)"),
            hoverinfo="skip",
            showlegend=True,
            name=f"Comfort Zone ({lower_bound:.1f}–{upper_bound:.1f}°C)",
        )
    )

    # 2. Desired Temperature Reference Line (Natural Green Dashed)
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=[desired_temp] * len(steps),
            mode="lines",
            line=dict(color=p["target"], width=2.0, dash="dash"),
            name=f"Desired Temp ({desired_temp:.1f}°C)",
            hovertemplate="Target: %{y:.1f}°C<extra></extra>",
        )
    )

    # 3. Outside Ambient Temperature
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=utility_df["outside_temp"],
            mode="lines",
            line=dict(color=p["outside"], width=1.6, dash="dot"),
            name="Outside Ambient",
            hovertemplate="Step %{x}: Outside %{y:.1f}°C<extra></extra>",
        )
    )

    # 4. Utility-Based Agent Temperature (Forest Green)
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=utility_df["actual_temp"],
            mode="lines+markers",
            line=dict(color=p["util_agent"], width=3.0),
            marker=dict(size=5.5, color=p["primary_green"]),
            name="Utility-Based Agent",
            text=utility_df["action"] if "action" in utility_df.columns else None,
            hovertemplate="Step %{x}: Utility %{y:.2f}°C (%{text})<extra></extra>",
        )
    )

    # 5. Simple Reflex Agent Temperature (Muted Teal)
    fig.add_trace(
        go.Scatter(
            x=reflex_df["step"],
            y=reflex_df["actual_temp"],
            mode="lines+markers",
            line=dict(color=p["reflex_agent"], width=2.8, dash="solid"),
            marker=dict(size=5.5, color=p["reflex_agent"], symbol="diamond"),
            name="Simple Reflex Thermostat",
            text=reflex_df["action"] if "action" in reflex_df.columns else None,
            hovertemplate="Step %{x}: Reflex %{y:.2f}°C (%{text})<extra></extra>",
        )
    )

    _apply_standard_chart_layout(
        fig=fig,
        title_text="<b>1. Synchronized Room Temperature Trajectory</b>",
        dark_mode=dark_mode,
        xaxis_title="Simulation Time Step",
        yaxis_title="Temperature (°C)",
        height=450,
        show_legend=True,
        hovermode="x unified",
        margin_top=110,
        margin_bottom=70,
        margin_left=75,
        margin_right=35,
    )
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
    - Utility-Based Agent: Forest Green (#245C43 Light / #75B58A Dark)
    - Simple Reflex Agent: Muted Teal (#4D9285 Light / #5BB5B5 Dark)
    """
    p = _get_theme_palette(dark_mode)

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
        fig.update_layout(title="No energy data recorded yet.", height=380)
        return fig

    fig.add_trace(
        go.Scatter(
            x=utility_df["step"],
            y=utility_df["cumulative_energy"],
            mode="lines+markers",
            line=dict(color=p["util_agent"], width=3.0),
            marker=dict(size=5.5),
            name="Utility-Based Agent",
            customdata=utility_df["energy_consumed"] if "energy_consumed" in utility_df.columns else [0.0] * len(utility_df),
            hovertemplate="Step %{x}: Cumulative %{y:.1f} units (Step: %{customdata:.1f})<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=reflex_df["step"],
            y=reflex_df["cumulative_energy"],
            mode="lines+markers",
            line=dict(color=p["reflex_agent"], width=2.8),
            marker=dict(size=5.5, symbol="diamond"),
            name="Simple Reflex Thermostat",
            customdata=reflex_df["energy_consumed"] if "energy_consumed" in reflex_df.columns else [0.0] * len(reflex_df),
            hovertemplate="Step %{x}: Cumulative %{y:.1f} units (Step: %{customdata:.1f})<extra></extra>",
        )
    )

    _apply_standard_chart_layout(
        fig=fig,
        title_text="<b>2. Cumulative Energy Consumption Over Time</b>",
        dark_mode=dark_mode,
        xaxis_title="Simulation Time Step",
        yaxis_title="Cumulative Energy (units)",
        height=390,
        show_legend=True,
        hovermode="x unified",
        margin_top=105,
        margin_bottom=70,
        margin_left=75,
        margin_right=35,
    )
    fig.update_yaxes(rangemode="tozero")
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
    - Utility-Based Agent: Forest Green (#245C43 Light / #75B58A Dark)
    - Simple Reflex Agent: Muted Teal (#4D9285 Light / #5BB5B5 Dark)
    """
    p = _get_theme_palette(dark_mode)

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
        fig.update_layout(title="No temperature error data recorded yet.", height=380)
        return fig

    fig.add_trace(
        go.Scatter(
            x=utility_df["step"],
            y=utility_df["abs_error"],
            mode="lines+markers",
            line=dict(color=p["util_agent"], width=2.8),
            marker=dict(size=5.5),
            name="Utility-Based Agent Error",
            hovertemplate="Step %{x}: |Error| = %{y:.2f}°C<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=reflex_df["step"],
            y=reflex_df["abs_error"],
            mode="lines+markers",
            line=dict(color=p["reflex_agent"], width=2.6),
            marker=dict(size=5.5, symbol="diamond"),
            name="Simple Reflex Thermostat Error",
            hovertemplate="Step %{x}: |Error| = %{y:.2f}°C<extra></extra>",
        )
    )

    _apply_standard_chart_layout(
        fig=fig,
        title_text="<b>3. Absolute Temperature Error Over Time (|T - T_target|)</b>",
        dark_mode=dark_mode,
        xaxis_title="Simulation Time Step",
        yaxis_title="Absolute Error (°C)",
        height=390,
        show_legend=True,
        hovermode="x unified",
        margin_top=105,
        margin_bottom=70,
        margin_left=75,
        margin_right=35,
    )
    fig.update_yaxes(rangemode="tozero")
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
    - Utility-Based Agent: Forest Green (#245C43 Light / #75B58A Dark)
    - Simple Reflex Agent: Muted Teal (#4D9285 Light / #5BB5B5 Dark)
    """
    p = _get_theme_palette(dark_mode)

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
    labels = ["HEAT", "COOL", "OFF"]

    u_counts = [int((u_active["action"] == a).sum()) if not u_active.empty else 0 for a in actions]
    r_counts = [int((r_active["action"] == a).sum()) if not r_active.empty else 0 for a in actions]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Utility-Based Agent",
            x=labels,
            y=u_counts,
            marker_color=p["util_agent"],
            marker_line=dict(color=p["border_color"], width=1),
            text=[str(c) for c in u_counts],
            textposition="auto",
            textfont=dict(size=13, weight="bold"),
        )
    )
    fig.add_trace(
        go.Bar(
            name="Simple Reflex Thermostat",
            x=labels,
            y=r_counts,
            marker_color=p["reflex_agent"],
            marker_line=dict(color=p["border_color"], width=1),
            text=[str(c) for c in r_counts],
            textposition="auto",
            textfont=dict(size=13, weight="bold"),
        )
    )

    _apply_standard_chart_layout(
        fig=fig,
        title_text="<b>4. Action Distribution Comparison (HEAT / COOL / OFF)</b>",
        dark_mode=dark_mode,
        yaxis_title="Number of Steps",
        height=380,
        show_legend=True,
        hovermode=None,
        margin_top=105,
        margin_bottom=65,
        margin_left=75,
        margin_right=35,
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(rangemode="tozero")
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
    - Utility-Based Agent: Forest Green (#245C43 Light / #75B58A Dark)
    - Simple Reflex Agent: Muted Teal (#4D9285 Light / #5BB5B5 Dark)
    """
    p = _get_theme_palette(dark_mode)
    categories = ["Cold", "Comfortable", "Hot"]
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
            marker_color=p["util_agent"],
            marker_line=dict(color=p["border_color"], width=1),
            text=[f"{v:.1f}%" for v in u_pcts],
            textposition="auto",
            textfont=dict(size=13, weight="bold"),
        )
    )
    fig.add_trace(
        go.Bar(
            name="Simple Reflex Thermostat",
            x=categories,
            y=r_pcts,
            marker_color=p["reflex_agent"],
            marker_line=dict(color=p["border_color"], width=1),
            text=[f"{v:.1f}%" for v in r_pcts],
            textposition="auto",
            textfont=dict(size=13, weight="bold"),
        )
    )

    _apply_standard_chart_layout(
        fig=fig,
        title_text="<b>5. Environment Classification Time Share (%)</b>",
        dark_mode=dark_mode,
        yaxis_title="Percentage of Steps (%)",
        height=380,
        show_legend=True,
        hovermode=None,
        margin_top=105,
        margin_bottom=65,
        margin_left=75,
        margin_right=35,
    )
    fig.update_layout(barmode="group", yaxis_range=[0, 108])
    return fig


# =========================================================
# Live Room Environment Simulation & Device Status Panel
# =========================================================

def get_live_room_simulation_html(
    current_temp: float,
    desired_temp: float,
    outside_temp: float,
    classification: str,
    action: str,
    agent_name: str = "Utility-Based Agent",
    fan_enabled: bool = True,
    window_open: bool = False,
    comfort_tolerance: float = 1.5,
    dark_mode: bool = False,
) -> str:
    """
    Renders the interactive 2D illustrated architectural room cross-section
    for the Live Room Environment Simulator, with animated thermal equipment:
    - Wall-mounted AC unit (Cool airflow streamlines & snowflake drift when COOL)
    - Electric room heater (Glowing elements & rising heat waves when HEAT)
    - Ceiling fan (Rotating blades, circulating airflow loops & indicator)
    - Window with curtains (Open/closed states, billowing drapes & outdoor breeze)
    - Digital wall thermostat HUD (Real-time live readouts)
    - Biophilic living plant and responsive ambient atmospheric wash.
    """
    p = _get_theme_palette(dark_mode)
    
    heater_on = (action == ACTION_HEAT)
    ac_on = (action == ACTION_COOL)
    fan_on = fan_enabled
    win_open = window_open

    # Room interior colors
    wall_bg = "#16241D" if dark_mode else "#F6F9F5"
    wall_border = "#2A3F33" if dark_mode else "#D9E3D6"
    wall_panel_line = "#1F3328" if dark_mode else "#E9EFE7"
    ceiling_bg = "#111C16" if dark_mode else "#EDF3EB"
    floor_bg = "#1D2821" if dark_mode else "#E6DEC8"
    floor_plank = "#25342B" if dark_mode else "#D9CFB8"
    baseboard_bg = "#142019" if dark_mode else "#D5CCB4"

    # Outdoor sky view depending on outdoor temperature
    if outside_temp < 18.0:
        sky_grad_start = "#607D8B"
        sky_grad_end = "#B0BEC5"
        tree_leaf_color = "#455A64"
        sun_cloud_svg = '<path d="M 625,115 Q 640,105 655,115 Q 670,110 680,122 Q 675,135 650,135 Q 625,135 625,115 Z" fill="#ECEFF1" opacity="0.85"/>'
        outdoor_desc = "Cold Winter Environment"
    elif outside_temp > 26.0:
        sky_grad_start = "#FF9800"
        sky_grad_end = "#FFE082"
        tree_leaf_color = "#33691E"
        sun_cloud_svg = '<circle cx="660" cy="115" r="18" fill="#FFB300" opacity="0.9"/><circle cx="660" cy="115" r="24" fill="#FFE082" opacity="0.35"/>'
        outdoor_desc = "Hot Summer Environment"
    else:
        sky_grad_start = "#0288D1"
        sky_grad_end = "#B3E5FC"
        tree_leaf_color = "#2E7D32"
        sun_cloud_svg = '<circle cx="670" cy="110" r="14" fill="#FDD835" opacity="0.9"/><path d="M 605,125 Q 620,115 635,125 Q 650,120 660,130 Q 655,140 630,140 Q 605,140 605,125 Z" fill="#FFFFFF" opacity="0.9"/>'
        outdoor_desc = "Mild Temperate Environment"

    # Wall Thermostat values
    temp_diff = current_temp - desired_temp
    if classification == CLASS_COLD:
        thermo_temp_color = "#38BDF8"
        thermo_badge_bg = "#162732" if dark_mode else "#EBF3F7"
        thermo_badge_color = "#6BA4C8" if dark_mode else "#2A6282"
        thermo_icon = "❄"
    elif classification == CLASS_HOT:
        thermo_temp_color = "#FB923C"
        thermo_badge_bg = "#2E2216" if dark_mode else "#FDF4E7"
        thermo_badge_color = "#E8A25B" if dark_mode else "#965C1E"
        thermo_icon = "🔥"
    else:
        thermo_temp_color = "#4ADE80" if dark_mode else "#16A34A"
        thermo_badge_bg = "#1B2D22" if dark_mode else "#EAF4EC"
        thermo_badge_color = "#75B58A" if dark_mode else "#245C43"
        thermo_icon = "🌿"

    # Mode banner texts and styling
    if action == ACTION_HEAT:
        banner_class = "heating"
        banner_icon = "🔥"
        banner_title = "HEATING ACTIVE"
        banner_msg = "Room heater: ON · Electric heating elements generating warm thermal airflow"
        ambient_tint = '<rect x="0" y="45" width="920" height="345" fill="url(#heatAmbientGrad)" pointer-events="none"/>'
    elif action == ACTION_COOL:
        banner_class = "cooling"
        banner_icon = "❄️"
        banner_title = "COOLING ACTIVE"
        banner_msg = "Air conditioner: ON · Refrigerant compressor delivering cold air currents"
        ambient_tint = '<rect x="0" y="45" width="920" height="345" fill="url(#coolAmbientGrad)" pointer-events="none"/>'
    else:
        banner_class = "idle"
        banner_icon = "⚡"
        banner_title = "SYSTEM IDLE"
        banner_msg = "Heating: OFF · Cooling: OFF · Room in natural thermal equilibrium"
        ambient_tint = ''

    # Fan visuals
    fan_spin_class = "fan-spin" if fan_on else ""
    fan_led_color = "#10B981" if fan_on else "#64748B"
    fan_blade_color = "#94A3B8" if dark_mode else "#64748B"

    # Window visuals
    curtain_class = "curtain-flutter" if win_open else ""
    curtain_color = "#2D4737" if dark_mode else "#CAD9CB"
    curtain_shade = "#1D3124" if dark_mode else "#B0C4B2"

    html = f"""<div class="room-sim-container">
<div class="room-sim-card">
<div class="room-sim-topbar">
<div class="room-agent-badge">
<span>🎮 Controlling Agent:</span>
<span class="agent-pill">{agent_name}</span>
</div>
<div class="room-sim-readouts">
<span class="room-pill primary"><strong>Room: {current_temp:.1f}°C</strong></span>
<span class="room-pill">Target: {desired_temp:.1f}°C (±{comfort_tolerance:.1f}°)</span>
<span class="room-pill">Outside: {outside_temp:.1f}°C</span>
<span class="room-pill">Mode: {action}</span>
</div>
</div>

<div class="room-status-banner {banner_class}">
<span class="status-dot {'orange' if action == ACTION_HEAT else ('blue' if action == ACTION_COOL else 'grey')}"></span>
<span><strong>{banner_title}</strong> — {banner_msg}</span>
</div>

<div class="room-sim-svg-wrap">
<svg viewBox="0 0 920 480" width="100%" height="auto" style="border-radius: 8px; border: 1px solid {p['border_color']}; background: {wall_bg}; display: block;" xmlns="http://www.w3.org/2000/svg">
<defs>
<!-- Gradients -->
<linearGradient id="wallShade" x1="0%" y1="0%" x2="0%" y2="100%">
<stop offset="0%" stop-color="{ceiling_bg}" stop-opacity="0.8"/>
<stop offset="100%" stop-color="{wall_bg}" stop-opacity="1"/>
</linearGradient>

<linearGradient id="floorGrad" x1="0%" y1="0%" x2="0%" y2="100%">
<stop offset="0%" stop-color="{baseboard_bg}"/>
<stop offset="15%" stop-color="{floor_bg}"/>
<stop offset="100%" stop-color="{'#131B16' if dark_mode else '#D0C6B0'}"/>
</linearGradient>

<linearGradient id="skyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
<stop offset="0%" stop-color="{sky_grad_start}"/>
<stop offset="100%" stop-color="{sky_grad_end}"/>
</linearGradient>

<linearGradient id="coolAirGrad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#00D2FF" stop-opacity="0.9"/>
<stop offset="100%" stop-color="#38BDF8" stop-opacity="0.1"/>
</linearGradient>

<linearGradient id="heatAirGrad" x1="0%" y1="100%" x2="0%" y2="0%">
<stop offset="0%" stop-color="#FF5722" stop-opacity="0.85"/>
<stop offset="70%" stop-color="#FFA000" stop-opacity="0.5"/>
<stop offset="100%" stop-color="#FFD54F" stop-opacity="0.05"/>
</linearGradient>

<radialGradient id="heatAmbientGrad" cx="800" cy="300" r="450" gradientUnits="userSpaceOnUse">
<stop offset="0%" stop-color="#FF6B35" stop-opacity="0.18"/>
<stop offset="60%" stop-color="#FF8C42" stop-opacity="0.06"/>
<stop offset="100%" stop-color="#FF8C42" stop-opacity="0"/>
</radialGradient>

<radialGradient id="coolAmbientGrad" cx="180" cy="100" r="450" gradientUnits="userSpaceOnUse">
<stop offset="0%" stop-color="#00B4D8" stop-opacity="0.18"/>
<stop offset="60%" stop-color="#38BDF8" stop-opacity="0.06"/>
<stop offset="100%" stop-color="#38BDF8" stop-opacity="0"/>
</radialGradient>

<clipPath id="windowClip">
<rect x="540" y="80" width="195" height="215" rx="4"/>
</clipPath>
</defs>

<!-- 1. Ceiling & Molding -->
<rect x="0" y="0" width="920" height="45" fill="{ceiling_bg}"/>
<line x1="0" y1="45" x2="920" y2="45" stroke="{wall_border}" stroke-width="1.8"/>

<!-- 2. Back Wall & Architecture Panels -->
<rect x="0" y="45" width="920" height="345" fill="url(#wallShade)"/>
<line x1="330" y1="45" x2="330" y2="390" stroke="{wall_panel_line}" stroke-width="1" stroke-dasharray="4,6"/>
<line x1="510" y1="45" x2="510" y2="390" stroke="{wall_panel_line}" stroke-width="1" stroke-dasharray="4,6"/>

<!-- 3. Baseboard and Floor -->
<rect x="0" y="380" width="920" height="14" fill="{baseboard_bg}"/>
<line x1="0" y1="380" x2="920" y2="380" stroke="{wall_border}" stroke-width="1.5"/>
<line x1="0" y1="394" x2="920" y2="394" stroke="{wall_border}" stroke-width="1"/>
<rect x="0" y="394" width="920" height="86" fill="url(#floorGrad)"/>

<!-- Floor Parquet / Plank Perspective Lines -->
<line x1="0" y1="420" x2="920" y2="420" stroke="{floor_plank}" stroke-width="1" opacity="0.6"/>
<line x1="0" y1="450" x2="920" y2="450" stroke="{floor_plank}" stroke-width="1" opacity="0.6"/>
<line x1="120" y1="394" x2="90" y2="480" stroke="{floor_plank}" stroke-width="1" opacity="0.4"/>
<line x1="300" y1="394" x2="270" y2="480" stroke="{floor_plank}" stroke-width="1" opacity="0.4"/>
<line x1="480" y1="394" x2="460" y2="480" stroke="{floor_plank}" stroke-width="1" opacity="0.4"/>
<line x1="660" y1="394" x2="650" y2="480" stroke="{floor_plank}" stroke-width="1" opacity="0.4"/>
<line x1="820" y1="394" x2="810" y2="480" stroke="{floor_plank}" stroke-width="1" opacity="0.4"/>

<!-- 4. Window & Outdoor Scenery -->
<g id="window-group">
<!-- Window Frame Outer Trim -->
<rect x="532" y="72" width="211" height="231" rx="6" fill="{'#101C16' if dark_mode else '#E2EBE0'}" stroke="{wall_border}" stroke-width="2"/>

<!-- Outdoor Landscape (Clipped inside window) -->
<g clip-path="url(#windowClip)">
<rect x="540" y="80" width="195" height="215" fill="url(#skyGrad)"/>
{sun_cloud_svg}
<!-- Tree Foliage & Ground Silhouette -->
<path d="M 535,275 Q 590,260 640,268 Q 690,275 745,265 L 745,300 L 535,300 Z" fill="{'#0F2418' if dark_mode else '#4B7354'}"/>
<path d="M 550,265 C 550,225 580,215 595,230 C 610,210 635,215 640,240 C 660,230 685,245 680,270 Z" fill="{tree_leaf_color}" opacity="0.95"/>
</g>

<!-- Window Panes & Opening State -->
{'<rect x="540" y="80" width="100" height="215" fill="none" stroke="' + wall_border + '" stroke-width="3"/><rect x="635" y="80" width="100" height="215" fill="none" stroke="' + wall_border + '" stroke-width="3"/><line x1="640" y1="80" x2="640" y2="295" stroke="' + wall_border + '" stroke-width="2.5"/><line x1="540" y1="185" x2="735" y2="185" stroke="' + wall_border + '" stroke-width="2.5"/>' if not win_open else
'<rect x="540" y="80" width="50" height="215" fill="rgba(255,255,255,0.12)" stroke="' + wall_border + '" stroke-width="2.5"/><line x1="590" y1="80" x2="590" y2="295" stroke="' + wall_border + '" stroke-width="3"/><rect x="640" y="80" width="95" height="215" fill="none" stroke="' + wall_border + '" stroke-width="2.5"/>'
}

<!-- Curtains & Drapery -->
<rect x="522" y="66" width="231" height="8" rx="3" fill="{'#3B5243' if dark_mode else '#9EADA0'}"/>
<g class="{curtain_class}">
<path d="M 522,74 Q 542,180 535,305 L 555,305 Q 562,180 545,74 Z" fill="{curtain_color}"/>
<path d="M 533,74 Q 547,180 543,305" stroke="{curtain_shade}" stroke-width="1.8" fill="none"/>
</g>
<g class="{curtain_class}" style="animation-delay: 0.3s;">
<path d="M 730,74 Q 722,180 735,305 L 755,305 Q 742,180 753,74 Z" fill="{curtain_color}"/>
<path d="M 743,74 Q 733,180 745,305" stroke="{curtain_shade}" stroke-width="1.8" fill="none"/>
</g>

<!-- Fresh Air Streamers Entering Window when Open -->
{'''
<g class="curtain-flutter">
<path d="M 590,140 C 560,170 510,195 440,210" fill="none" stroke="#52B788" stroke-width="3" stroke-linecap="round" stroke-dasharray="8 4" opacity="0.8"/>
<path d="M 590,175 C 550,200 480,220 410,230" fill="none" stroke="#75B58A" stroke-width="2.5" stroke-linecap="round" stroke-dasharray="8 4" opacity="0.8"/>
<text x="595" y="130" font-size="11" font-weight="700" fill="#2D8053">FRESH AIR INFLOW</text>
</g>
''' if win_open else ''}
</g>

<!-- 5. Wall-Mounted Air Conditioner (Left Wall) -->
<g id="ac-unit-group">
<!-- AC Housing -->
<rect x="65" y="65" width="225" height="72" rx="8" fill="{'#1A2B22' if dark_mode else '#FFFFFF'}" stroke="{wall_border}" stroke-width="1.8" filter="drop-shadow(0 4px 6px rgba(0,0,0,0.08))"/>
<!-- Top Intake Grille Lines -->
<line x1="85" y1="78" x2="270" y2="78" stroke="{'#2B4034' if dark_mode else '#E2E8E0'}" stroke-width="2"/>
<line x1="85" y1="84" x2="270" y2="84" stroke="{'#2B4034' if dark_mode else '#E2E8E0'}" stroke-width="2"/>
<!-- Bottom Bevel / Seam -->
<line x1="65" y1="118" x2="290" y2="118" stroke="{wall_border}" stroke-width="1.2"/>
<!-- Brand Label -->
<text x="85" y="104" font-size="10" font-weight="700" fill="{p['text_muted']}" letter-spacing="0.05em">ECO-INVERTER AC</text>

<!-- AC Digital Status Indicator & LED -->
<circle cx="265" cy="101" r="4.5" fill="{'#00D2FF' if ac_on else '#64748B'}" filter="{'drop-shadow(0 0 6px #00D2FF)' if ac_on else 'none'}"/>
<text x="248" y="104" font-size="9" font-weight="700" fill="{'#00D2FF' if ac_on else p['text_muted']}">{'ON' if ac_on else 'OFF'}</text>

<!-- Vent Flap -->
{'<rect x="75" y="122" width="205" height="10" rx="3" fill="#0284C7" transform="rotate(12 75 122)"/>' if ac_on else
'<rect x="75" y="122" width="205" height="7" rx="2" fill="' + ('#23382C' if dark_mode else '#D0DCD0') + '"/>'
}

<!-- Cool Airflow Streamlines & Snowflake Particles -->
{'''
<g class="cool-airflow-group">
<path class="cool-stream cool-stream-1" d="M 120,135 C 160,200 230,265 330,300" fill="none" stroke="url(#coolAirGrad)" stroke-width="4.5" stroke-linecap="round"/>
<path class="cool-stream cool-stream-2" d="M 170,135 C 220,195 290,250 400,285" fill="none" stroke="url(#coolAirGrad)" stroke-width="3.8" stroke-linecap="round"/>
<path class="cool-stream cool-stream-3" d="M 220,135 C 270,185 350,235 460,265" fill="none" stroke="url(#coolAirGrad)" stroke-width="3" stroke-linecap="round"/>
<text class="snowflake sf-1" x="180" y="210" font-size="15" fill="#38BDF8">❄</text>
<text class="snowflake sf-2" x="260" y="260" font-size="17" fill="#7DD3FC">❄</text>
<text class="snowflake sf-3" x="340" y="230" font-size="13" fill="#BAE6FD">❄</text>
<text class="snowflake sf-4" x="410" y="290" font-size="15" fill="#38BDF8">❄</text>
</g>
''' if ac_on else ''}
</g>

<!-- 6. Ceiling Fan (Top Center) -->
<g id="ceiling-fan-group">
<!-- Ceiling Mount Plate -->
<rect x="444" y="40" width="32" height="7" rx="2" fill="{'#253A2D' if dark_mode else '#94A3B8'}"/>
<!-- Downrod -->
<rect x="457" y="47" width="6" height="38" fill="{'#3B5243' if dark_mode else '#64748B'}"/>

<!-- Motor Hub Body -->
<circle cx="460" cy="85" r="18" fill="{'#1E3126' if dark_mode else '#CBD5E1'}" stroke="{wall_border}" stroke-width="1.5"/>
<circle cx="460" cy="85" r="5" fill="{fan_led_color}" filter="{'drop-shadow(0 0 5px #10B981)' if fan_on else 'none'}"/>

<!-- 4 Fan Blades -->
<g class="fan-blades-group {fan_spin_class}">
<!-- Blade 1: Left -->
<path d="M 460,85 C 445,68 375,64 350,78 C 375,92 445,95 460,85 Z" fill="{fan_blade_color}"/>
<!-- Blade 2: Right -->
<path d="M 460,85 C 475,68 545,64 570,78 C 545,92 475,95 460,85 Z" fill="{fan_blade_color}"/>
<!-- Blade 3: Top -->
<path d="M 460,85 C 442,70 438,0 452,-20 C 466,0 472,70 460,85 Z" fill="{fan_blade_color}"/>
<!-- Blade 4: Bottom -->
<path d="M 460,85 C 442,100 438,170 452,190 C 466,170 472,100 460,85 Z" fill="{fan_blade_color}"/>
</g>

<!-- Circulating Airflow Curves below Fan -->
{'''
<g opacity="0.8">
<path d="M 410,125 Q 460,155 510,125" fill="none" stroke="' + p["sage_green"] + '" stroke-width="2.5" stroke-dasharray="6,4"/>
<path d="M 385,150 Q 460,190 535,150" fill="none" stroke="' + p["sage_green"] + '" stroke-width="2" stroke-dasharray="6,4"/>
</g>
''' if fan_on else ''}
</g>

<!-- 7. Electric Room Heater (Floor Lower Right) -->
<g id="heater-group">
<!-- Base Stand -->
<rect x="745" y="360" width="105" height="28" rx="6" fill="{'#1A2820' if dark_mode else '#94A3B8'}" stroke="{wall_border}" stroke-width="1.5"/>

<!-- Heater Tower Body -->
<rect x="755" y="225" width="85" height="145" rx="8" fill="{'#203227' if dark_mode else '#F1F5F0'}" stroke="{wall_border}" stroke-width="2" filter="drop-shadow(0 4px 8px rgba(0,0,0,0.12))"/>

<!-- Heater Brand & Controls -->
<rect x="765" y="235" width="65" height="16" rx="4" fill="{'#15221B' if dark_mode else '#E2EBE0'}"/>
<circle cx="818" cy="243" r="3.5" fill="{'#FF5722' if heater_on else '#64748B'}" filter="{'drop-shadow(0 0 6px #FF5722)' if heater_on else 'none'}"/>
<text x="772" y="246" font-size="8" font-weight="700" fill="{'#FF5722' if heater_on else p['text_muted']}">{'HEAT ON' if heater_on else 'HEAT OFF'}</text>

<!-- Vertical Heating Grille Elements -->
{'''
<!-- Active Glowing Heating Elements -->
<rect x="770" y="260" width="10" height="98" rx="3" fill="#FF5722" filter="drop-shadow(0 0 6px #FF5722)"/>
<rect x="792" y="260" width="10" height="98" rx="3" fill="#FFA000" filter="drop-shadow(0 0 8px #FFA000)"/>
<rect x="814" y="260" width="10" height="98" rx="3" fill="#FF5722" filter="drop-shadow(0 0 6px #FF5722)"/>

<!-- Rising Warm Heat Waves -->
<g class="heat-wave-group">
<path class="heat-wave hw-1" d="M 770,220 Q 750,170 770,120 T 760,70" fill="none" stroke="url(#heatAirGrad)" stroke-width="4" stroke-linecap="round"/>
<path class="heat-wave hw-2" d="M 797,220 Q 815,165 795,110 T 805,60" fill="none" stroke="url(#heatAirGrad)" stroke-width="4.5" stroke-linecap="round"/>
<path class="heat-wave hw-3" d="M 825,220 Q 845,175 825,130 T 835,80" fill="none" stroke="url(#heatAirGrad)" stroke-width="3.5" stroke-linecap="round"/>
</g>
''' if heater_on else '''
<!-- Inactive Dormant Heating Grilles -->
<rect x="770" y="260" width="10" height="98" rx="3" fill="''' + ('#2D4234' if dark_mode else '#CBD5E1') + '''"/>
<rect x="792" y="260" width="10" height="98" rx="3" fill="''' + ('#2D4234' if dark_mode else '#CBD5E1') + '''"/>
<rect x="814" y="260" width="10" height="98" rx="3" fill="''' + ('#2D4234' if dark_mode else '#CBD5E1') + '''"/>
'''}
</g>

<!-- 8. Digital Wall Thermostat HUD (Lower Left Wall) -->
<g id="thermostat-hud">
<!-- Mounting Glass Frame -->
<rect x="75" y="195" width="180" height="135" rx="10" fill="{'#17261E' if dark_mode else '#FFFFFF'}" stroke="{wall_border}" stroke-width="2" filter="drop-shadow(0 4px 10px rgba(0,0,0,0.1))"/>
<!-- Internal Display Bezel -->
<rect x="85" y="205" width="160" height="115" rx="7" fill="{'#101A15' if dark_mode else '#F0F6EE'}"/>

<!-- Temperature Large Readout -->
<text x="96" y="244" font-size="28" font-weight="800" fill="{thermo_temp_color}" font-family="system-ui, -apple-system, sans-serif">{current_temp:.1f}°C</text>

<!-- Classification / Condition Pill -->
<rect x="96" y="254" width="70" height="18" rx="4" fill="{thermo_badge_bg}" stroke="{thermo_badge_color}" stroke-width="1"/>
<text x="103" y="267" font-size="10.5" font-weight="700" fill="{thermo_badge_color}">{classification.upper()}</text>

<!-- Mode Icon & Label -->
<text x="175" y="267" font-size="11" font-weight="700" fill="{p['text_secondary']}">{action}</text>

<!-- Small Target & Outdoor Details -->
<text x="96" y="291" font-size="10.5" font-weight="500" fill="{p['text_muted']}">Target: {desired_temp:.1f}°C (±{comfort_tolerance:.1f}°)</text>
<text x="96" y="307" font-size="10.5" font-weight="500" fill="{p['text_muted']}">Outdoor: {outside_temp:.1f}°C</text>
</g>

<!-- 9. Biophilic Living Plant (Floor Left) -->
<g id="biophilic-plant">
<!-- Ceramic Planter Pot -->
<path d="M 42,390 L 52,350 L 82,350 L 92,390 Z" fill="{'#2A3F33' if dark_mode else '#D0DCCF'}" stroke="{wall_border}" stroke-width="1.5"/>
<ellipse cx="67" cy="350" rx="15" ry="3.5" fill="{'#192820' if dark_mode else '#9FB3A0'}"/>
<!-- Plant Stems and Leaves -->
<path d="M 67,350 Q 55,305 35,290 Q 55,300 67,350 Z" fill="{'#4E875B' if dark_mode else '#2D6A4F'}"/>
<path d="M 67,350 Q 75,295 95,275 Q 85,300 67,350 Z" fill="{'#5F9E6D' if dark_mode else '#40916C'}"/>
<path d="M 67,350 Q 67,270 65,245 Q 73,275 67,350 Z" fill="{'#75B58A' if dark_mode else '#52B788'}"/>
<path d="M 67,350 Q 85,320 110,315 Q 92,330 67,350 Z" fill="{'#3A6D47' if dark_mode else '#1B4332'}"/>
</g>

<!-- 10. Ambient Atmospheric Gradient Overlay -->
{ambient_tint}

</svg>
</div>
</div>
</div>"""
    return _clean_html(html)


def get_device_status_panel_html(
    action: str,
    fan_enabled: bool,
    window_open: bool,
    dark_mode: bool = False,
) -> str:
    """
    Renders a compact, high-contrast device monitoring panel displaying
    the operational status of the 4 simulated room devices:
    - Room Heater (ON/OFF)
    - Air Conditioner (ON/OFF)
    - Ceiling Fan (ON/OFF)
    - Window (OPEN/CLOSED)
    """
    p = _get_theme_palette(dark_mode)

    heater_on = (action == ACTION_HEAT)
    ac_on = (action == ACTION_COOL)
    fan_on = fan_enabled
    win_open = window_open

    # Helper for status pill styling
    def _dev_card(name: str, status_text: str, is_active: bool, active_color: str, active_bg: str, active_border: str, icon_svg: str, subtext: str) -> str:
        bg = active_bg if is_active else p["card_bg"]
        border = active_border if is_active else p["border_color"]
        badge_bg = active_color if is_active else ("#2A3830" if dark_mode else "#E5EBE3")
        badge_color = "#FFFFFF" if is_active else p["text_muted"]
        title_color = active_color if is_active else p["text_primary"]

        return f"""<div class="dev-card" style="background: {bg}; border: 1px solid {border};">
<div class="dev-header">
<div class="dev-icon" style="color: {active_color if is_active else p['text_muted']};">
{icon_svg}
</div>
<span class="dev-badge" style="background: {badge_bg}; color: {badge_color};">
{status_text}
</span>
</div>
<div class="dev-name" style="color: {title_color};">{name}</div>
<div class="dev-subtext" style="color: {p['text_muted']};">{subtext}</div>
</div>"""

    # Device 1: Heater
    heater_icon = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4M8 6h8M6 10h12M6 14h12M6 18h12M9 22h6"/></svg>"""
    heater_html = _dev_card(
        name="Room Heater",
        status_text="ON" if heater_on else "OFF",
        is_active=heater_on,
        active_color="#E76F51",
        active_bg="#2E1E12" if dark_mode else "#FDF3EB",
        active_border="#5E3516" if dark_mode else "#F5C299",
        icon_svg=heater_icon,
        subtext="Active Heating" if heater_on else "Standby (Off)",
    )

    # Device 2: Air Conditioner
    ac_icon = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="12" x="2" y="4" rx="2"/><path d="M6 16v2M18 16v2M6 12h12"/></svg>"""
    ac_html = _dev_card(
        name="Air Conditioner",
        status_text="ON" if ac_on else "OFF",
        is_active=ac_on,
        active_color="#0284C7",
        active_bg="#12232E" if dark_mode else "#EBF6FD",
        active_border="#1B475E" if dark_mode else "#B8DEF5",
        icon_svg=ac_icon,
        subtext="Active Cooling" if ac_on else "Standby (Off)",
    )

    # Device 3: Ceiling Fan
    fan_icon = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 12c-2-2.5-5.5-2.5-7.5-.5S2.5 17 4.5 19s5.5 2.5 7.5.5"/><path d="M12 12c2.5-2 2.5-5.5.5-7.5S7 2.5 5 4.5s-2.5 5.5-.5 7.5"/><path d="M12 12c2 2.5 5.5 2.5 7.5.5s2-5.5 0-7.5-5.5-2.5-7.5-.5"/><circle cx="12" cy="12" r="2"/></svg>"""
    fan_html = _dev_card(
        name="Ceiling Fan",
        status_text="ON" if fan_on else "OFF",
        is_active=fan_on,
        active_color="#059669",
        active_bg="#132B1E" if dark_mode else "#EAF6EE",
        active_border="#235237" if dark_mode else "#B6E2C5",
        icon_svg=fan_icon,
        subtext="Air Circulation Active" if fan_on else "Stationary (Off)",
    )

    # Device 4: Window
    window_icon = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 12h18M12 3v18"/></svg>"""
    window_html = _dev_card(
        name="Window Ventilation",
        status_text="OPEN" if win_open else "CLOSED",
        is_active=win_open,
        active_color="#16A34A",
        active_bg="#142B1A" if dark_mode else "#EDF8EF",
        active_border="#275432" if dark_mode else "#BCE3C4",
        icon_svg=window_icon,
        subtext="Natural Fresh Airflow" if win_open else "Sealed Glazing",
    )

    panel_html = f"""<div class="device-panel-grid">
{heater_html}
{ac_html}
{fan_html}
{window_html}
</div>"""
    return _clean_html(panel_html)
