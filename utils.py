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
    Ensure every line of generated HTML is stripped of leading indentation.
    This guarantees Streamlit's Markdown parser will NEVER treat HTML tags
    as indented code blocks.
    """
    return "\n".join(line.lstrip() for line in html.strip().splitlines())


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
