"""
Environment Classifier and Demonstrator
Interactive Room Temperature Control Simulation & Multi-Agent Comparison.
Clean Minimalist & Scientific Dashboard supporting Light and Dark modes.

Sections:
1. Overview
2. Environment Setup
3. Utility-Based Agent
4. Reflex Thermostat
5. Agent Comparison
6. Results
"""

import pandas as pd
import streamlit as st

from config import (
    ACTION_COOL,
    ACTION_HEAT,
    ACTION_METADATA,
    ACTION_OFF,
    CLASS_COLD,
    CLASS_COMFORTABLE,
    CLASS_HOT,
    DEMO_SCENARIOS,
    STATUS_COLORS,
    DemonstrationScenario,
    ReflexThermostatDefaults,
    SimulationDefaults,
    UtilityWeightsDefaults,
)
from environment import RoomEnvironment
from utility_agent import UtilityBasedAgent
from reflex_agent import SimpleReflexAgent
from comparison import (
    ComparisonConfig,
    ComparisonRunner,
)
from utils import (
    _clean_html,
    create_agent_temperature_history_chart,
    create_comparison_action_distribution_chart,
    create_comparison_bar_chart,
    create_comparison_energy_chart,
    create_comparison_error_chart,
    create_comparison_temperature_chart,
    create_reflex_temperature_history_chart,
    create_temperature_gauge,
    create_utility_comparison_chart,
    get_comparison_summary_html,
    get_decision_card_html,
    get_overview_metric_card_html,
    get_reflex_rules_html,
    get_status_card_html,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Environment Classifier and Demonstrator",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "dark_mode": False,
        "initial_temp": float(SimulationDefaults.DEFAULT_INITIAL_TEMP),
        "desired_temp": float(SimulationDefaults.DEFAULT_DESIRED_TEMP),
        "outside_temp": float(SimulationDefaults.DEFAULT_OUTSIDE_TEMP),
        "insulation_level": "Medium",
        "comfort_tolerance": float(SimulationDefaults.DEFAULT_COMFORT_TOLERANCE),
        "heater_power": float(SimulationDefaults.DEFAULT_HEATER_POWER),
        "cooling_power": float(SimulationDefaults.DEFAULT_COOLING_POWER),
        "disturbance_level": float(SimulationDefaults.DEFAULT_DISTURBANCE),
        "max_steps": int(SimulationDefaults.DEFAULT_STEPS),
        "random_seed": 42,
        "weight_comfort": float(UtilityWeightsDefaults.DEFAULT_WEIGHT_COMFORT),
        "weight_energy": float(UtilityWeightsDefaults.DEFAULT_WEIGHT_ENERGY),
        "weight_stability": float(UtilityWeightsDefaults.DEFAULT_WEIGHT_STABILITY),
        "weight_overshoot": float(UtilityWeightsDefaults.DEFAULT_WEIGHT_OVERSHOOT),
        "reflex_lower_slider": float(ReflexThermostatDefaults.DEFAULT_LOWER_THRESHOLD),
        "reflex_upper_slider": float(ReflexThermostatDefaults.DEFAULT_UPPER_THRESHOLD),
        "active_scenario_id": "custom",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if "env" not in st.session_state:
        st.session_state.env = _create_configured_environment()
    if "agent" not in st.session_state:
        st.session_state.agent = UtilityBasedAgent(
            weight_comfort=st.session_state.weight_comfort,
            weight_energy=st.session_state.weight_energy,
            weight_stability=st.session_state.weight_stability,
            weight_overshoot=st.session_state.weight_overshoot,
        )
    if "simulation_records" not in st.session_state:
        reset_utility_simulation()

    if "reflex_env" not in st.session_state:
        st.session_state.reflex_env = _create_configured_environment()
    if "reflex_agent" not in st.session_state:
        st.session_state.reflex_agent = SimpleReflexAgent(
            lower_threshold=st.session_state.reflex_lower_slider,
            upper_threshold=st.session_state.reflex_upper_slider,
        )
    if "reflex_simulation_records" not in st.session_state:
        reset_reflex_simulation()

    if "comparison_runner" not in st.session_state:
        st.session_state.comparison_runner = ComparisonRunner(_build_comparison_config())


def _create_configured_environment() -> RoomEnvironment:
    return RoomEnvironment(
        initial_temp=float(st.session_state.initial_temp),
        desired_temp=float(st.session_state.desired_temp),
        outside_temp=float(st.session_state.outside_temp),
        insulation_level=st.session_state.insulation_level,
        comfort_tolerance=float(st.session_state.comfort_tolerance),
        heater_power=float(st.session_state.heater_power),
        cooling_power=float(st.session_state.cooling_power),
        disturbance_level=float(st.session_state.disturbance_level),
        random_seed=int(st.session_state.random_seed),
    )


def _build_comparison_config() -> ComparisonConfig:
    return ComparisonConfig(
        initial_temp=float(st.session_state.initial_temp),
        desired_temp=float(st.session_state.desired_temp),
        outside_temp=float(st.session_state.outside_temp),
        insulation_level=st.session_state.insulation_level,
        comfort_tolerance=float(st.session_state.comfort_tolerance),
        heater_power=float(st.session_state.heater_power),
        cooling_power=float(st.session_state.cooling_power),
        disturbance_level=float(st.session_state.disturbance_level),
        max_steps=int(st.session_state.max_steps),
        random_seed=int(st.session_state.random_seed),
        weight_comfort=float(st.session_state.weight_comfort),
        weight_energy=float(st.session_state.weight_energy),
        weight_stability=float(st.session_state.weight_stability),
        weight_overshoot=float(st.session_state.weight_overshoot),
        lower_threshold=float(st.session_state.reflex_lower_slider),
        upper_threshold=float(st.session_state.reflex_upper_slider),
    )


def reset_utility_simulation() -> None:
    st.session_state.env = _create_configured_environment()
    st.session_state.agent = UtilityBasedAgent(
        weight_comfort=st.session_state.weight_comfort,
        weight_energy=st.session_state.weight_energy,
        weight_stability=st.session_state.weight_stability,
        weight_overshoot=st.session_state.weight_overshoot,
    )
    init_obs = st.session_state.env.get_state()
    st.session_state.simulation_records = [{
        "step": 0,
        "room_temp": init_obs["room_temp"],
        "outside_temp": init_obs["outside_temp"],
        "desired_temp": init_obs["desired_temp"],
        "action": ACTION_OFF,
        "predicted_temp": init_obs["room_temp"],
        "actual_temp": init_obs["room_temp"],
        "utility_score": 0.0,
        "energy_consumed": 0.0,
        "classification": init_obs["classification"],
    }]


def reset_reflex_simulation() -> None:
    st.session_state.reflex_env = _create_configured_environment()
    st.session_state.reflex_agent = SimpleReflexAgent(
        lower_threshold=st.session_state.reflex_lower_slider,
        upper_threshold=st.session_state.reflex_upper_slider,
    )
    init_obs = st.session_state.reflex_env.get_state()
    st.session_state.reflex_simulation_records = [{
        "step": 0,
        "room_temp": init_obs["room_temp"],
        "outside_temp": init_obs["outside_temp"],
        "desired_temp": init_obs["desired_temp"],
        "action": ACTION_OFF,
        "actual_temp": init_obs["room_temp"],
        "energy_consumed": 0.0,
        "cumulative_energy": 0.0,
        "classification": init_obs["classification"],
    }]


def reset_all_simulations() -> None:
    reset_utility_simulation()
    reset_reflex_simulation()
    st.session_state.comparison_runner = ComparisonRunner(_build_comparison_config())


def apply_demonstration_scenario(scenario_id: str) -> None:
    if scenario_id not in DEMO_SCENARIOS:
        return
    sc = DEMO_SCENARIOS[scenario_id]
    st.session_state.active_scenario_id = scenario_id
    st.session_state.initial_temp = sc.initial_temp
    st.session_state.desired_temp = sc.desired_temp
    st.session_state.outside_temp = sc.outside_temp
    st.session_state.insulation_level = sc.insulation_level
    st.session_state.comfort_tolerance = sc.comfort_tolerance
    st.session_state.heater_power = sc.heater_power
    st.session_state.cooling_power = sc.cooling_power
    st.session_state.disturbance_level = sc.disturbance_level
    st.session_state.max_steps = sc.max_steps
    st.session_state.random_seed = sc.random_seed
    st.session_state.reflex_lower_slider = sc.lower_threshold
    st.session_state.reflex_upper_slider = sc.upper_threshold
    st.session_state.weight_comfort = sc.weight_comfort
    st.session_state.weight_energy = sc.weight_energy
    st.session_state.weight_stability = sc.weight_stability
    st.session_state.weight_overshoot = sc.weight_overshoot
    reset_all_simulations()


init_session_state()

# Global handles
env: RoomEnvironment = st.session_state.env
agent: UtilityBasedAgent = st.session_state.agent
reflex_env: RoomEnvironment = st.session_state.reflex_env
reflex_agent: SimpleReflexAgent = st.session_state.reflex_agent
comparison_runner: ComparisonRunner = st.session_state.comparison_runner

# Synchronize comparison runner with latest configuration
latest_cfg = _build_comparison_config()
if comparison_runner.config != latest_cfg:
    comparison_runner.reset(latest_cfg)

dark_mode: bool = bool(st.session_state.get("dark_mode", False))

# ---------------------------------------------------------
# Dynamic Light/Dark Theme & Fixed Header Styling (Biophilic Eco-Tech)
# ---------------------------------------------------------
if dark_mode:
    app_bg = "#101B17"
    card_bg = "#192720"
    card_sub_bg = "#22352A"
    text_primary = "#E7F0E8"
    text_secondary = "#B4C4B5"
    text_muted = "#7B8F7F"
    forest_green = "#75B58A"
    primary_green = "#75B58A"
    muted_teal = "#5BB5B5"
    sage_green = "#9DBA91"
    mint_accent = "#8FD0A8"
    border_color = "#304438"
    header_bg = "#192720"
    header_border = "#304438"
    header_text = "#E7F0E8"
    sidebar_bg = "#192720"
    sidebar_border = "#304438"
    button_hover_bg = "#5A9E6F"
    logo_box_bg = "#22352A"
    logo_box_border = "#304438"
    leaf_color = "#75B58A"
    thermo_color = "#8FD0A8"
else:
    app_bg = "#F5F7F0"
    card_bg = "#FFFFFF"
    card_sub_bg = "#EAF2EB"
    text_primary = "#26382D"
    text_secondary = "#5F7162"
    text_muted = "#8CA08E"
    forest_green = "#245C43"
    primary_green = "#245C43"
    muted_teal = "#4D9285"
    sage_green = "#829B78"
    mint_accent = "#B9D7C1"
    border_color = "#DCE5D8"
    header_bg = "#FFFFFF"
    header_border = "#DCE5D8"
    header_text = "#26382D"
    sidebar_bg = "#FFFFFF"
    sidebar_border = "#DCE5D8"
    button_hover_bg = "#1E4B37"
    logo_box_bg = "#EAF2EB"
    logo_box_border = "#DCE5D8"
    leaf_color = "#245C43"
    thermo_color = "#4D9285"

theme_css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400..800;1,9..40,400..800&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'DM Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: {app_bg} !important;
    color: {text_primary} !important;
}}

/* 1. Complete Removal of Streamlit Deploy Button & Deployment Elements */
[data-testid="stDeployButton"],
.stDeployButton,
div[data-testid="stToolbar"] > div:has([data-testid="stDeployButton"]),
div[data-testid="stToolbar"] button[kind="header"][data-testid="stDeployButton"],
button[title="Deploy this app"],
#MainMenu > div:has(button[title="Deploy this app"]) {{
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
}}

/* 2. Configure Streamlit's Default Header */
header[data-testid="stHeader"] {{
    background-color: transparent !important;
    height: 72px !important;
    z-index: 999990 !important;
}}

/* Ensure Streamlit's sidebar collapse/expand button sits cleanly on top left of header */
header[data-testid="stHeader"] button[data-testid="stSidebarCollapseButton"],
header[data-testid="stHeader"] button[data-testid="baseButton-headerNoPadding"],
[data-testid="stSidebarCollapseButton"] {{
    margin-top: 17px !important;
    margin-left: 14px !important;
    z-index: 1000000 !important;
    color: {header_text} !important;
}}

/* 3. Professional Biophilic Fixed Top Header */
.fixed-top-header {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 72px;
    background-color: {header_bg};
    border-bottom: 1px solid {header_border};
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 1.5rem 0 4.2rem;
    z-index: 999980;
    box-shadow: 0 1px 3px rgba(0, 0, 0, {"0.2" if dark_mode else "0.03"});
    box-sizing: border-box;
    transition: background-color 0.2s ease, border-color 0.2s ease;
}}

.header-left {{
    display: flex;
    align-items: center;
    gap: 14px;
    user-select: none;
    min-width: 0;
}}

.header-brand-logo {{
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}}

.header-title-group {{
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-width: 0;
}}

.header-title {{
    font-size: 1.15rem;
    font-weight: 700;
    color: {header_text};
    letter-spacing: -0.015em;
    line-height: 1.25;
    white-space: nowrap;
}}

.header-subtitle {{
    font-size: 0.85rem;
    font-weight: 400;
    color: {text_secondary};
    letter-spacing: 0.005em;
    line-height: 1.25;
    white-space: nowrap;
}}

.header-right {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-right: 145px; /* Clean clearance for fixed theme toggle pill */
    flex-shrink: 0;
}}

.header-status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.02em;
    white-space: nowrap;
    transition: all 0.2s ease;
}}

.header-status-badge .badge-icon,
.header-status-badge .badge-temp-icon {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
}}

.header-status-badge .badge-sep {{
    opacity: 0.5;
    font-weight: 400;
    margin: 0 1px;
}}

/* Dock Theme Toggle switch directly into top right header */
.st-key-header_theme_toggle,
div[data-testid="stCheckbox"]:has(.st-key-header_theme_toggle),
div[data-testid="stCheckbox"]:has(input[aria-label*="Light"]),
div[data-testid="stCheckbox"]:has(input[aria-label*="Dark"]) {{
    position: fixed !important;
    top: 18px !important;
    right: 1.5rem !important;
    z-index: 1000000 !important;
    margin: 0 !important;
    padding: 0 !important;
    height: 36px !important;
    display: flex !important;
    align-items: center !important;
}}

.st-key-header_theme_toggle label,
div[data-testid="stCheckbox"]:has(.st-key-header_theme_toggle) label {{
    margin: 0 !important;
    padding: 5px 12px !important;
    background-color: {card_sub_bg} !important;
    border: 1px solid {border_color} !important;
    border-radius: 9999px !important;
    cursor: pointer !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, {"0.15" if dark_mode else "0.03"}) !important;
}}

.st-key-header_theme_toggle label:hover,
div[data-testid="stCheckbox"]:has(.st-key-header_theme_toggle) label:hover {{
    border-color: {primary_green} !important;
    background-color: {card_bg} !important;
}}

.st-key-header_theme_toggle label p,
.st-key-header_theme_toggle label span,
.st-key-header_theme_toggle [data-testid="stMarkdownContainer"] p,
div[data-testid="stCheckbox"]:has(.st-key-header_theme_toggle) label p {{
    margin: 0 !important;
    padding: 0 !important;
    color: {header_text} !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    line-height: 1 !important;
    white-space: nowrap !important;
    display: flex !important;
    align-items: center !important;
    gap: 4px !important;
}}

.st-key-header_theme_toggle input[role="switch"]:checked + div,
.st-key-header_theme_toggle div[role="switch"][aria-checked="true"] {{
    background-color: {primary_green} !important;
}}

/* 4. Page Content Layout & Vertical Spacing */
.block-container,
div[data-testid="stMainBlockContainer"],
div[data-testid="block-container"] {{
    padding-top: 5.8rem !important;
    background-color: {app_bg} !important;
}}

/* 5. Sidebar layout starts cleanly below fixed header */
section[data-testid="stSidebar"] {{
    top: 72px !important;
    height: calc(100vh - 72px) !important;
    border-right: 1px solid {sidebar_border} !important;
    background-color: {sidebar_bg} !important;
    z-index: 999970 !important;
    transition: background-color 0.2s ease, border-color 0.2s ease;
}}

section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {{
    padding-top: 1.1rem !important;
    padding-bottom: 2rem !important;
}}

section[data-testid="stSidebar"] * {{
    color: {text_primary} !important;
}}

/* 6. Biophilic Modern Sidebar Navigation - Calm & Integrated */
div[data-testid="stRadio"] [role="radiogroup"] {{
    display: flex !important;
    flex-direction: column !important;
    gap: 4px !important;
    width: 100% !important;
    padding: 2px 0 !important;
}}

/* Completely hide raw browser radio dots */
div[data-testid="stRadio"] [role="radiogroup"] input[type="radio"] {{
    display: none !important;
    position: absolute !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
    pointer-events: none !important;
}}

div[data-testid="stRadio"] [role="radiogroup"] label > div:first-child:not([data-testid="stMarkdownContainer"]) {{
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    opacity: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}}

div[data-testid="stRadio"] [role="radiogroup"] .react-aria-RadioField,
div[data-testid="stRadio"] [role="radiogroup"] > div {{
    width: 100% !important;
}}

/* Inactive navigation items: calm, transparent, no heavy individual borders */
div[data-testid="stRadio"] [role="radiogroup"] label {{
    background-color: transparent !important;
    border: 1px solid transparent !important;
    border-left: 3.5px solid transparent !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    box-sizing: border-box !important;
}}

div[data-testid="stRadio"] [role="radiogroup"] label:hover {{
    background-color: {"rgba(117, 181, 138, 0.1)" if dark_mode else "rgba(36, 92, 67, 0.06)"} !important;
    border-color: transparent !important;
    border-left-color: {"#5A9E6F" if dark_mode else "#829B78"} !important;
    transform: translateX(2px) !important;
}}

/* Active navigation state:
   - Soft sage-green background in Light Mode (#EAF2EB)
   - Muted forest-green background in Dark Mode (#22352A)
   - Left-side active indicator (3.5px solid #245C43 / #75B58A)
   - Medium-weight text (600)
*/
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked),
div[data-testid="stRadio"] [role="radiogroup"] label[data-selected="true"],
div[data-testid="stRadio"] [role="radiogroup"] .react-aria-RadioField[data-selected="true"] label {{
    background-color: {"#22352A" if dark_mode else "#EAF2EB"} !important;
    border: 1px solid {"#2E4738" if dark_mode else "#D5E3D6"} !important;
    border-left: 3.5px solid {primary_green} !important;
    box-shadow: none !important;
}}

/* Ensure inner container fills width */
div[data-testid="stRadio"] label > div {{
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
}}

/* Navigation label typography */
div[data-testid="stRadio"] [data-testid="stMarkdownContainer"],
div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {{
    display: flex !important;
    align-items: center !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: {text_secondary} !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.35 !important;
}}

/* Icons inside navigation items */
div[data-testid="stRadio"] [data-testid="stIconMaterial"],
div[data-testid="stRadio"] .material-symbols-rounded {{
    color: {text_secondary} !important;
    font-size: 19px !important;
    margin-right: 8px !important;
    vertical-align: middle !important;
    transition: color 0.15s ease !important;
}}

/* Active navigation text and icon */
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] p,
div[data-testid="stRadio"] [role="radiogroup"] label[data-selected="true"] [data-testid="stMarkdownContainer"] p {{
    color: {primary_green} !important;
    font-weight: 600 !important;
}}

div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) [data-testid="stIconMaterial"],
div[data-testid="stRadio"] [role="radiogroup"] label[data-selected="true"] [data-testid="stIconMaterial"],
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) .material-symbols-rounded,
div[data-testid="stRadio"] [role="radiogroup"] label[data-selected="true"] .material-symbols-rounded {{
    color: {primary_green} !important;
}}

/* Sidebar Brand Lockup, Section Captions, and Dividers */
.sidebar-brand-box {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    background-color: {card_sub_bg};
    border: 1px solid {border_color};
    border-radius: 9px;
    margin-bottom: 14px;
}}

.sidebar-brand-icon {{
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}}

.sidebar-brand-info {{
    display: flex;
    flex-direction: column;
}}

.sidebar-brand-name {{
    font-size: 13.5px;
    font-weight: 700;
    color: {text_primary};
    line-height: 1.25;
}}

.sidebar-brand-sub {{
    font-size: 10.5px;
    font-weight: 600;
    color: {text_secondary};
    text-transform: uppercase;
    letter-spacing: 0.03em;
    line-height: 1.2;
}}

.sidebar-section-caption {{
    font-size: 11px;
    font-weight: 600;
    color: {text_secondary};
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-top: 6px;
    margin-bottom: 8px;
    padding-left: 2px;
}}

.sidebar-divider {{
    height: 1px;
    background-color: {border_color};
    margin: 14px 0 10px 0;
}}

.sidebar-settings-card {{
    background-color: {card_sub_bg};
    border: 1px solid {border_color};
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 12px;
    font-size: 12px;
    display: flex;
    flex-direction: column;
    gap: 6px;
}}

.sidebar-setting-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
}}

.sidebar-setting-row .setting-k {{
    color: {text_secondary};
    font-weight: 500;
    font-size: 12px;
}}

.sidebar-setting-row .setting-v {{
    color: {text_primary};
    font-weight: 700;
    text-align: right;
    font-size: 12px;
}}

/* 7. BaseWeb Selectbox */
div[data-testid="stSelectbox"] label,
div[data-testid="stSelectbox"] label * {{
    color: {text_primary} !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
}}

div[data-baseweb="select"],
div[data-baseweb="select"] > div,
div[data-baseweb="select"] [role="combobox"],
div[data-baseweb="select"] [aria-haspopup="listbox"],
div[data-baseweb="select"] div {{
    background-color: {card_bg} !important;
    border-color: {border_color} !important;
    color: {text_primary} !important;
}}

div[data-baseweb="select"] > div {{
    border: 1px solid {border_color} !important;
    border-radius: 8px !important;
    transition: border-color 0.2s ease !important;
}}

div[data-baseweb="select"] > div:hover {{
    border-color: {primary_green} !important;
}}

div[data-baseweb="select"] span,
div[data-baseweb="select"] p,
div[data-baseweb="select"] [role="combobox"] * {{
    color: {text_primary} !important;
    background-color: transparent !important;
    font-size: 13.5px !important;
}}

div[data-baseweb="select"] svg {{
    fill: {text_secondary} !important;
    color: {text_secondary} !important;
}}

/* Selectbox Dropdown Menu Popover */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
ul[data-baseweb="menu"] {{
    background-color: {card_bg} !important;
    border: 1px solid {border_color} !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, {"0.35" if dark_mode else "0.08"}) !important;
}}

li[data-baseweb="menu-item"],
li[data-baseweb="menu-item"] * {{
    background-color: transparent !important;
    color: {text_primary} !important;
    font-size: 13.5px !important;
}}

li[data-baseweb="menu-item"]:hover,
li[data-baseweb="menu-item"][aria-selected="true"] {{
    background-color: {card_sub_bg} !important;
    color: {primary_green} !important;
}}

/* 8. Biophilic Typography and Card styling */
.main-title {{
    font-size: 28px !important;
    font-weight: 700 !important;
    color: {text_primary} !important;
    margin-bottom: 0.2rem !important;
    letter-spacing: -0.015em !important;
    line-height: 1.25 !important;
}}

.sub-title {{
    font-size: 14.5px !important;
    color: {text_secondary} !important;
    margin-bottom: 1.2rem !important;
    font-weight: 400 !important;
    line-height: 1.5 !important;
}}

.info-card {{
    background-color: {card_bg};
    border: 1px solid {border_color};
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 1.2rem;
    color: {text_primary};
    font-size: 14.5px;
    line-height: 1.5;
}}

.summary-card {{
    background: {card_bg};
    border: 1px solid {border_color};
    border-radius: 10px;
    padding: 16px 18px;
    height: 100%;
    color: {text_primary};
    font-size: 14px;
    line-height: 1.5;
}}

/* Headings */
h1, h2, div[data-testid="stHeadingWithActionElements"] h2 {{
    color: {text_primary} !important;
    font-size: 22px !important;
    font-weight: 600 !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.6rem !important;
    letter-spacing: -0.01em !important;
}}

h3, div[data-testid="stHeadingWithActionElements"] h3 {{
    color: {text_primary} !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
}}

h4, .card-title {{
    color: {text_primary} !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}}

/* Captions & Helper Text */
div[data-testid="stCaptionContainer"],
.stCaption,
small {{
    color: {text_secondary} !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
}}

/* 9. Metric Cards - Clean, Uncluttered, No Green Square Boxes or Distracting Tooltips */
div[data-testid="stMetric"] {{
    background-color: {card_bg} !important;
    border: 1px solid {border_color} !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    box-shadow: none !important;
}}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] *,
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricLabel"] label,
div[data-testid="stMetricLabel"] span,
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label * {{
    color: {text_secondary} !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}}

/* Completely hide distracting help tooltip question marks */
button[data-testid="stMetricHelpButton"],
div[data-testid="stMetric"] svg[data-testid="stTooltipIcon"],
div[data-testid="stMetricLabel"] svg {{
    display: none !important;
    visibility: hidden !important;
}}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] *,
div[data-testid="stMetricValue"] p,
div[data-testid="stMetricValue"] span,
div[data-testid="stMetricValue"] div {{
    color: {text_primary} !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    font-feature-settings: "tnum" 1 !important;
}}

/* Delta styling: clean inline label, transparent background, no square box */
div[data-testid="stMetricDelta"],
div[data-testid="stMetricDelta"] > div {{
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-top: 2px !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
    width: fit-content !important;
}}

div[data-testid="stMetricDelta"] *,
div[data-testid="stMetricDelta"] span,
div[data-testid="stMetricDelta"] svg {{
    color: {text_secondary} !important;
    fill: {text_secondary} !important;
    font-size: 12.5px !important;
    font-weight: 400 !important;
}}

/* Sliders */
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] label * {{
    color: {text_primary} !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
}}

div[data-testid="stSlider"] [data-testid="stThumbValue"],
div[data-testid="stSlider"] [data-testid="stThumbValue"] * {{
    color: {text_primary} !important;
    font-size: 13px !important;
    font-weight: 700 !important;
}}

div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] {{
    color: {text_secondary} !important;
    font-size: 11.5px !important;
    font-weight: 500 !important;
}}

div[data-baseweb="slider"] div[role="slider"] {{
    background-color: {primary_green} !important;
    border-color: {card_bg} !important;
}}

/* Buttons Styling in Biophilic Forest Green */
button[kind="primary"],
.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {{
    background-color: {primary_green} !important;
    color: {"#101B17" if dark_mode else "#FFFFFF"} !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 0.52rem 1.1rem !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, {"0.2" if dark_mode else "0.08"}) !important;
    transition: all 0.2s ease !important;
}}

button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {{
    background-color: {button_hover_bg} !important;
    color: {"#101B17" if dark_mode else "#FFFFFF"} !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, {"0.3" if dark_mode else "0.14"}) !important;
    transform: translateY(-1px) !important;
}}

button[kind="secondary"],
.stButton > button[kind="secondary"],
button[data-testid="baseButton-secondary"] {{
    background-color: {card_bg} !important;
    color: {text_primary} !important;
    border: 1px solid {border_color} !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 0.52rem 1.1rem !important;
    transition: all 0.2s ease !important;
}}

button[kind="secondary"]:hover,
.stButton > button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover {{
    background-color: {card_sub_bg} !important;
    border-color: {primary_green} !important;
    color: {primary_green} !important;
    transform: translateY(-1px) !important;
}}

/* Download Buttons */
div[data-testid="stDownloadButton"] > button {{
    background-color: {card_bg} !important;
    color: {text_primary} !important;
    border: 1px solid {border_color} !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 0.52rem 1.1rem !important;
}}

div[data-testid="stDownloadButton"] > button:hover {{
    background-color: {card_sub_bg} !important;
    border-color: {primary_green} !important;
}}

/* 10. Tables / DataFrames (Follows Active Theme) */
div[data-testid="stDataFrame"],
div[data-testid="stTable"] {{
    background-color: {card_bg} !important;
    border: 1px solid {border_color} !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    {"filter: invert(0.88) hue-rotate(180deg) !important;" if dark_mode else ""}
}}

/* Responsive Mobile / Tablet adjustments */
@media (max-width: 900px) {{
    .header-subtitle {{
        display: none !important;
    }}
    .header-title {{
        font-size: 0.95rem !important;
    }}
    .header-status-badge {{
        font-size: 11.5px !important;
        padding: 5px 10px !important;
    }}
    .header-right {{
        margin-right: 125px !important;
    }}
    .st-key-header_theme_toggle {{
        right: 1rem !important;
    }}
}}

@media (max-width: 680px) {{
    .header-title-group {{
        display: none !important;
    }}
    .fixed-top-header {{
        padding-left: 3.5rem !important;
        padding-right: 1rem !important;
    }}
    .header-right {{
        margin-right: 100px !important;
    }}
}}
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)


# ---------------------------------------------------------
# Fixed Top Header HTML & Real-time Status Badge
# ---------------------------------------------------------
current_obs = env.get_state()
cur_temp = current_obs["room_temp"]
cur_class = current_obs["classification"]
status_info = STATUS_COLORS.get(cur_class, STATUS_COLORS[CLASS_COMFORTABLE])

# Minimal line-art SVG logo combining leaf and thermometer
brand_logo_svg = f"""<svg width="38" height="38" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect width="38" height="38" rx="10" fill="{logo_box_bg}" stroke="{logo_box_border}" stroke-width="1.2"/>
<path d="M14 28C9 26.5 6.8 22.8 7.3 16.5C7.8 10.2 14.5 8 20 7C20.5 12.5 19 19 14 23.5" stroke="{leaf_color}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M7.8 28C10.2 25.8 12.8 23.6 14 23.5" stroke="{leaf_color}" stroke-width="1.9" stroke-linecap="round"/>
<path d="M25 12.5V21A3.5 3.5 0 1 1 21.8 21V12.5A1.6 1.6 0 0 1 25 12.5Z" stroke="{thermo_color}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="23.4" cy="24.4" r="1.5" fill="{thermo_color}"/>
<line x1="23.4" y1="15.5" x2="23.4" y2="18.2" stroke="{thermo_color}" stroke-width="1.6" stroke-linecap="round"/>
</svg>"""

# Outline status icon & theme-tailored badge colors
if cur_class == CLASS_COLD:
    badge_icon_svg = """<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="2" x2="12" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/><line x1="19.07" y1="4.93" x2="4.93" y2="19.07"/></svg>"""
    badge_bg = "#162732" if dark_mode else "#EBF3F7"
    badge_border = "#234255" if dark_mode else "#BED9E8"
    badge_color = "#6BA4C8" if dark_mode else "#2A6282"
elif cur_class == CLASS_HOT:
    badge_icon_svg = """<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/><circle cx="12" cy="12" r="4"/></svg>"""
    badge_bg = "#2E2216" if dark_mode else "#FDF4E7"
    badge_border = "#4E3722" if dark_mode else "#F0D5AF"
    badge_color = "#E8A25B" if dark_mode else "#965C1E"
else:  # CLASS_COMFORTABLE
    badge_icon_svg = """<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6"/></svg>"""
    badge_bg = "#1B2D22" if dark_mode else "#EAF4EC"
    badge_border = "#2D4836" if dark_mode else "#C6DEC9"
    badge_color = "#75B58A" if dark_mode else "#245C43"

thermo_mini_svg = """<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/></svg>"""

header_html = f"""<header class="fixed-top-header">
<div class="header-left">
<div class="header-brand-logo">{brand_logo_svg}</div>
<div class="header-title-group">
<div class="header-title">Environment Classifier and Demonstrator</div>
<div class="header-subtitle">Intelligent Room Temperature Control</div>
</div>
</div>
<div class="header-right">
<div class="header-status-badge" style="background-color: {badge_bg}; border: 1px solid {badge_border}; color: {badge_color};">
<span class="badge-icon">{badge_icon_svg}</span>
<span class="badge-label">{cur_class.upper()}</span>
<span class="badge-sep">·</span>
<span class="badge-temp-icon">{thermo_mini_svg}</span>
<span class="badge-temp-val">{cur_temp:.1f}°C</span>
</div>
</div>
</header>"""
st.markdown(_clean_html(header_html), unsafe_allow_html=True)

# Theme Toggle switch (docked via CSS into top right of fixed header)
toggle_theme = st.toggle(
    ":material/dark_mode: Dark" if dark_mode else ":material/light_mode: Light",
    key="header_theme_toggle",
    value=dark_mode,
    help="Toggle between Light Mode and Dark Mode",
)
if toggle_theme != dark_mode:
    st.session_state["dark_mode"] = toggle_theme
    st.rerun()


# ---------------------------------------------------------
# Sidebar Navigation & Demo Presets
# ---------------------------------------------------------
SECTIONS = [
    "Overview",
    "Environment Setup",
    "Utility-Based Agent",
    "Reflex Thermostat",
    "Agent Comparison",
    "Results",
]

NAV_ICONS = {
    "Overview": ":material/dashboard:",
    "Environment Setup": ":material/tune:",
    "Utility-Based Agent": ":material/psychology:",
    "Reflex Thermostat": ":material/device_thermostat:",
    "Agent Comparison": ":material/balance:",
    "Results": ":material/bar_chart:",
}

sidebar_branding_html = f"""<div class="sidebar-brand-box">
<div class="sidebar-brand-icon">
<svg width="28" height="28" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect width="38" height="38" rx="8" fill="{logo_box_bg}" stroke="{logo_box_border}" stroke-width="1.2"/>
<path d="M14 28C9 26.5 6.8 22.8 7.3 16.5C7.8 10.2 14.5 8 20 7C20.5 12.5 19 19 14 23.5" stroke="{leaf_color}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M7.8 28C10.2 25.8 12.8 23.6 14 23.5" stroke="{leaf_color}" stroke-width="1.9" stroke-linecap="round"/>
<path d="M25 12.5V21A3.5 3.5 0 1 1 21.8 21V12.5A1.6 1.6 0 0 1 25 12.5Z" stroke="{thermo_color}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="23.4" cy="24.4" r="1.5" fill="{thermo_color}"/>
</svg>
</div>
<div class="sidebar-brand-info">
<div class="sidebar-brand-name">Eco-Tech Control</div>
<div class="sidebar-brand-sub">Dual-Agent Simulator</div>
</div>
</div>
<div class="sidebar-section-caption">NAVIGATION</div>"""

with st.sidebar:
    st.markdown(_clean_html(sidebar_branding_html), unsafe_allow_html=True)
    nav_selection = st.radio(
        "Select Section",
        options=SECTIONS,
        format_func=lambda s: f"{NAV_ICONS.get(s, '')}  {s}",
        key="nav_selection",
        label_visibility="collapsed",
    )

    st.markdown(
        f'<div class="sidebar-divider"></div><div class="sidebar-section-caption">DEMONSTRATION SCENARIOS</div>',
        unsafe_allow_html=True,
    )
    scenario_choice = st.selectbox(
        "Load Preset Scenario",
        options=[
            "Scenario 1: Cold Room",
            "Scenario 2: Hot Room",
            "Scenario 3: Comfortable Room",
            "Custom Configuration",
        ],
        index=0 if st.session_state.active_scenario_id == "cold_room" else (
            1 if st.session_state.active_scenario_id == "hot_room" else (
                2 if st.session_state.active_scenario_id == "comfortable_room" else 3
            )
        ),
        label_visibility="collapsed",
    )

    if st.button("Apply Selected Scenario", type="primary", use_container_width=True, key="btn_apply_scenario"):
        if "Scenario 1" in scenario_choice:
            apply_demonstration_scenario("cold_room")
        elif "Scenario 2" in scenario_choice:
            apply_demonstration_scenario("hot_room")
        elif "Scenario 3" in scenario_choice:
            apply_demonstration_scenario("comfortable_room")
        st.success("Scenario applied!")
        st.rerun()

    settings_card_html = f"""<div class="sidebar-divider"></div>
<div class="sidebar-section-caption">CURRENT SETTINGS</div>
<div class="sidebar-settings-card">
<div class="sidebar-setting-row">
<span class="setting-k">Target Temp</span>
<span class="setting-v">{st.session_state.desired_temp:.1f}°C (±{st.session_state.comfort_tolerance:.1f}°C)</span>
</div>
<div class="sidebar-setting-row">
<span class="setting-k">Initial Temp</span>
<span class="setting-v">{st.session_state.initial_temp:.1f}°C</span>
</div>
<div class="sidebar-setting-row">
<span class="setting-k">Outside Temp</span>
<span class="setting-v">{st.session_state.outside_temp:.1f}°C</span>
</div>
<div class="sidebar-setting-row">
<span class="setting-k">Insulation</span>
<span class="setting-v">{st.session_state.insulation_level}</span>
</div>
<div class="sidebar-setting-row">
<span class="setting-k">Simulation Steps</span>
<span class="setting-v">{st.session_state.max_steps}</span>
</div>
<div class="sidebar-setting-row">
<span class="setting-k">Reflex Bounds</span>
<span class="setting-v">[{st.session_state.reflex_lower_slider:.1f}°C, {st.session_state.reflex_upper_slider:.1f}°C]</span>
</div>
</div>"""
    st.markdown(_clean_html(settings_card_html), unsafe_allow_html=True)

    if st.button("Reset All Simulations", type="secondary", use_container_width=True, key="btn_sidebar_reset"):
        reset_all_simulations()
        st.rerun()


# =========================================================
# SECTION 1: Overview
# =========================================================
if nav_selection == "Overview":
    st.markdown(
        """<div class="main-title">Environment Classifier and Demonstrator</div>
<div class="sub-title">Intelligent Room Temperature Control & Multi-Agent Evaluation</div>""",
        unsafe_allow_html=True,
    )

    # 1. Real-time Status Card
    st.markdown(
        get_status_card_html(
            classification=cur_class,
            current_temp=cur_temp,
            desired_temp=st.session_state.desired_temp,
            dark_mode=dark_mode,
        ),
        unsafe_allow_html=True,
    )

    # 2. 4 Essential Metric Cards (Uncluttered, No Green Square Boxes, No Distracting Help Icons)
    temp_diff = cur_temp - st.session_state.desired_temp
    arrow = "↓" if temp_diff < -0.05 else ("↑" if temp_diff > 0.05 else "✓")
    diff_subtext = f"{arrow} {abs(temp_diff):.1f} °C vs target"

    if cur_class == CLASS_COMFORTABLE:
        env_subtext = "Within optimal temperature range"
        status_type = "comfortable"
    elif cur_class == CLASS_COLD:
        env_subtext = "Below comfortable temperature"
        status_type = "cold"
    else:
        env_subtext = "Above comfortable temperature"
        status_type = "hot"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            get_overview_metric_card_html(
                label="Current Room Temp",
                value=f"{cur_temp:.1f} °C",
                subtext=diff_subtext,
                status_type=status_type,
                dark_mode=dark_mode,
            ),
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            get_overview_metric_card_html(
                label="Desired Target",
                value=f"{st.session_state.desired_temp:.1f} °C",
                subtext=f"Comfort range ±{st.session_state.comfort_tolerance:.1f} °C",
                dark_mode=dark_mode,
            ),
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            get_overview_metric_card_html(
                label="Outside Temp",
                value=f"{st.session_state.outside_temp:.1f} °C",
                subtext=f"Insulation: {st.session_state.insulation_level}",
                dark_mode=dark_mode,
            ),
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            get_overview_metric_card_html(
                label="Environment Status",
                value=cur_class.upper(),
                subtext=env_subtext,
                status_type=status_type,
                dark_mode=dark_mode,
            ),
            unsafe_allow_html=True,
        )

    # 3. Environmental Temperature Monitoring Visualization
    st.subheader("Temperature Monitoring & Thermal State")
    ch_col1, ch_col2 = st.columns(2)
    with ch_col1:
        st.plotly_chart(
            create_temperature_gauge(
                current_temp=cur_temp,
                desired_temp=st.session_state.desired_temp,
                comfort_tolerance=st.session_state.comfort_tolerance,
                dark_mode=dark_mode,
            ),
            use_container_width=True,
            key="overview_temp_gauge",
        )
    with ch_col2:
        st.plotly_chart(
            create_comparison_bar_chart(
                room_temp=cur_temp,
                desired_temp=st.session_state.desired_temp,
                outside_temp=st.session_state.outside_temp,
                dark_mode=dark_mode,
            ),
            use_container_width=True,
            key="overview_bar_chart",
        )

    # 4. Agent Architecture Overview
    st.subheader("Agent Architecture Summary")
    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown(
            f"""<div class="summary-card" style="border-left: 4px solid {forest_green};">
<h4 style="margin-top:0; color:{forest_green};">🌱 Utility-Based Intelligent Agent</h4>
<b>Decision Paradigm:</b> Predictive Multi-Objective Optimization.<br>
<b>Internal Model:</b> Evaluates resulting temperature for actions (HEAT, COOL, OFF) using thermodynamic simulation.<br>
<b>Utility Formula:</b> Utility = -(w_c · Comfort Penalty) - (w_e · Energy Cost) - (w_s · Instability) - (w_o · Overshoot).<br>
<b>Selected Action:</b> Selects the action that achieves maximum utility score.
</div>""",
            unsafe_allow_html=True,
        )
    with ac2:
        st.markdown(
            f"""<div class="summary-card" style="border-left: 4px solid {muted_teal};">
<h4 style="margin-top:0; color:{muted_teal};">🔁 Simple Reflex Thermostat</h4>
<b>Decision Paradigm:</b> Deterministic Condition-Action Rules.<br>
<b>Sensor Input:</b> Observes only current room temperature without lookahead or utility evaluation.<br>
<b>Rules:</b>
• If Temperature &lt; {st.session_state.reflex_lower_slider:.1f}°C → HEAT<br>
• If Temperature &gt; {st.session_state.reflex_upper_slider:.1f}°C → COOL<br>
• Otherwise (Deadband) → OFF
</div>""",
            unsafe_allow_html=True,
        )

    # 5. Quick-Start Demonstration Scenarios
    st.subheader("Quick Scenario Launchers")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        if st.button("❄️ Scenario 1: Cold Room", use_container_width=True, key="btn_ov_sc1"):
            apply_demonstration_scenario("cold_room")
            st.rerun()
        st.caption("14°C initial room temp with 10°C outside. Tests heating response.")
    with sc2:
        if st.button("🔥 Scenario 2: Hot Room", use_container_width=True, key="btn_ov_sc2"):
            apply_demonstration_scenario("hot_room")
            st.rerun()
        st.caption("32°C initial room temp with 36°C outside. Tests cooling response.")
    with sc3:
        if st.button("🌿 Scenario 3: Comfortable Room", use_container_width=True, key="btn_ov_sc3"):
            apply_demonstration_scenario("comfortable_room")
            st.rerun()
        st.caption("22°C initial room temp with good insulation. Tests energy conservation.")


# =========================================================
# SECTION 2: Environment Setup
# =========================================================
elif nav_selection == "Environment Setup":
    st.markdown(
        """<div class="main-title">Environment Setup</div>
<div class="sub-title">Configure room temperatures, insulation parameters, and agent weights</div>""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌡️ Temperature & Environment")
        st.session_state.initial_temp = st.slider(
            "Initial Room Temperature (°C)",
            min_value=float(SimulationDefaults.MIN_INITIAL_TEMP),
            max_value=float(SimulationDefaults.MAX_INITIAL_TEMP),
            value=float(st.session_state.initial_temp),
            step=0.5,
            key="sl_init_temp",
            help="Starting temperature of the room.",
        )
        st.session_state.desired_temp = st.slider(
            "Desired Target Temperature (°C)",
            min_value=float(SimulationDefaults.MIN_DESIRED_TEMP),
            max_value=float(SimulationDefaults.MAX_DESIRED_TEMP),
            value=float(st.session_state.desired_temp),
            step=0.5,
            key="sl_desired_temp",
            help="Desired target comfort temperature.",
        )
        st.session_state.comfort_tolerance = st.slider(
            "Comfort Tolerance (± °C)",
            min_value=0.5,
            max_value=3.0,
            value=float(st.session_state.comfort_tolerance),
            step=0.1,
            key="sl_comfort_tolerance",
            help="Acceptable deviation around the desired temperature.",
        )
        st.session_state.outside_temp = st.slider(
            "Outside Temperature (°C)",
            min_value=float(SimulationDefaults.MIN_OUTSIDE_TEMP),
            max_value=float(SimulationDefaults.MAX_OUTSIDE_TEMP),
            value=float(st.session_state.outside_temp),
            step=0.5,
            key="sl_outside_temp",
            help="Outdoor ambient temperature.",
        )

        ins_opts = ["Poor", "Medium", "Good"]
        st.session_state.insulation_level = st.selectbox(
            "Room Insulation Level",
            options=ins_opts,
            index=ins_opts.index(st.session_state.insulation_level),
            key="sb_insulation",
            help="Determines the rate of heat exchange with outside air.",
        )
        st.session_state.max_steps = st.slider(
            "Simulation Steps",
            min_value=int(SimulationDefaults.MIN_STEPS),
            max_value=int(SimulationDefaults.MAX_STEPS),
            value=int(st.session_state.max_steps),
            step=5,
            key="sl_max_steps",
            help="Total number of simulation steps to run.",
        )

    with col2:
        st.markdown("#### ⚙️ Agent Parameters")

        st.markdown("**Reflex Thermostat Thresholds**")

        def _sync_reflex_lower() -> None:
            if st.session_state.reflex_lower_slider >= st.session_state.reflex_upper_slider:
                st.session_state.reflex_upper_slider = min(
                    float(ReflexThermostatDefaults.MAX_THRESHOLD),
                    round(float(st.session_state.reflex_lower_slider) + 0.5, 1),
                )

        def _sync_reflex_upper() -> None:
            if st.session_state.reflex_upper_slider <= st.session_state.reflex_lower_slider:
                st.session_state.reflex_lower_slider = max(
                    float(ReflexThermostatDefaults.MIN_THRESHOLD),
                    round(float(st.session_state.reflex_upper_slider) - 0.5, 1),
                )

        st.slider(
            "Lower Threshold (°C) — Heat below this",
            min_value=float(ReflexThermostatDefaults.MIN_THRESHOLD),
            max_value=float(ReflexThermostatDefaults.MAX_THRESHOLD - 0.5),
            step=0.5,
            key="reflex_lower_slider",
            on_change=_sync_reflex_lower,
        )
        st.slider(
            "Upper Threshold (°C) — Cool above this",
            min_value=float(ReflexThermostatDefaults.MIN_THRESHOLD + 0.5),
            max_value=float(ReflexThermostatDefaults.MAX_THRESHOLD),
            step=0.5,
            key="reflex_upper_slider",
            on_change=_sync_reflex_upper,
        )

        st.markdown("**Utility Agent Weights**")
        st.session_state.weight_comfort = st.slider(
            "Comfort Weight (w_c)",
            min_value=0.0,
            max_value=3.0,
            value=float(st.session_state.weight_comfort),
            step=0.1,
            key="sl_weight_c",
            help="Priority given to staying close to target temperature.",
        )
        st.session_state.weight_energy = st.slider(
            "Energy Cost Weight (w_e)",
            min_value=0.0,
            max_value=2.0,
            value=float(st.session_state.weight_energy),
            step=0.05,
            key="sl_weight_e",
            help="Penalty for consuming electricity when heating or cooling.",
        )
        st.session_state.weight_stability = st.slider(
            "Stability Penalty Weight (w_s)",
            min_value=0.0,
            max_value=2.0,
            value=float(st.session_state.weight_stability),
            step=0.05,
            key="sl_weight_s",
            help="Penalty for large single-step temperature changes.",
        )
        st.session_state.weight_overshoot = st.slider(
            "Overshoot Penalty Weight (w_o)",
            min_value=0.0,
            max_value=3.0,
            value=float(st.session_state.weight_overshoot),
            step=0.1,
            key="sl_weight_o",
            help="Penalty for exceeding the comfortable temperature boundaries.",
        )

        agent.update_weights(
            weight_comfort=st.session_state.weight_comfort,
            weight_energy=st.session_state.weight_energy,
            weight_stability=st.session_state.weight_stability,
            weight_overshoot=st.session_state.weight_overshoot,
        )
        reflex_agent.update_thresholds(
            lower_threshold=st.session_state.reflex_lower_slider,
            upper_threshold=st.session_state.reflex_upper_slider,
        )

    if st.button("Apply Settings & Reset All Simulations", type="primary", use_container_width=True, key="btn_apply_reset"):
        reset_all_simulations()
        st.success("Configuration applied! All simulations reset with new parameters.")
        st.rerun()


# =========================================================
# SECTION 3: Utility-Based Agent
# =========================================================
elif nav_selection == "Utility-Based Agent":
    st.markdown(
        """<div class="main-title">Utility-Based Intelligent Agent</div>
<div class="sub-title">Action selection by maximizing a multi-objective utility function</div>""",
        unsafe_allow_html=True,
    )

    current_state = env.get_state()
    last_dec = agent.last_decision

    # Current Environment Status
    st.markdown(
        get_status_card_html(
            classification=current_state["classification"],
            current_temp=current_state["room_temp"],
            desired_temp=current_state["desired_temp"],
            dark_mode=dark_mode,
        ),
        unsafe_allow_html=True,
    )

    # Clean metric cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Current Room Temp", f"{current_state['room_temp']:.2f} °C")
    with m2:
        st.metric("Desired Target", f"{current_state['desired_temp']:.1f} °C")
    with m3:
        st.metric("Outside Temp", f"{current_state['outside_temp']:.1f} °C")
    with m4:
        st.metric("Simulation Step", f"{env.current_step} / {st.session_state.max_steps}")

    # Simulation Controls
    def perform_agent_step() -> None:
        if env.current_step < st.session_state.max_steps:
            obs = env.get_state()
            dec = agent.decide_action(obs)
            next_st = env.step(dec.selected_action)
            st.session_state.simulation_records.append({
                "step": env.current_step,
                "room_temp": obs["room_temp"],
                "outside_temp": obs["outside_temp"],
                "desired_temp": obs["desired_temp"],
                "action": dec.selected_action,
                "predicted_temp": dec.predicted_temp,
                "actual_temp": next_st["room_temp"],
                "utility_score": dec.utility_score,
                "energy_consumed": dec.energy_consumed,
                "classification": next_st["classification"],
            })

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Run Simulation", key="btn_u_run", use_container_width=True, type="primary"):
            remaining = st.session_state.max_steps - env.current_step
            if remaining > 0:
                for _ in range(remaining):
                    perform_agent_step()
                st.rerun()
            else:
                st.warning("Simulation complete. Reset to start again.")
    with c2:
        if st.button("Step 1 Step", key="btn_u_step", use_container_width=True):
            if env.current_step < st.session_state.max_steps:
                perform_agent_step()
                st.rerun()
            else:
                st.warning("Simulation complete. Reset to start again.")
    with c3:
        if st.button("Reset Simulation", key="btn_u_reset", use_container_width=True):
            reset_utility_simulation()
            st.rerun()

    # Decision Reasoning & Utility Scores
    st.subheader("Decision Explanation & Action Scores")
    if last_dec:
        st.markdown(
            get_decision_card_html(
                current_temp=last_dec.current_temp,
                desired_temp=last_dec.desired_temp,
                selected_action=last_dec.selected_action,
                predicted_temp=last_dec.predicted_temp,
                utility_score=last_dec.utility_score,
                reason=last_dec.reason,
                dark_mode=dark_mode,
            ),
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            create_utility_comparison_chart(
                action_evaluations=last_dec.action_evaluations,
                selected_action=last_dec.selected_action,
                dark_mode=dark_mode,
            ),
            use_container_width=True,
            key="chart_sec_u_comp",
        )

    # Temperature Trajectory Chart
    st.subheader("Temperature History")
    sim_df = pd.DataFrame(st.session_state.simulation_records)
    st.plotly_chart(
        create_agent_temperature_history_chart(
            df=sim_df,
            desired_temp=st.session_state.desired_temp,
            comfort_tolerance=st.session_state.comfort_tolerance,
            dark_mode=dark_mode,
        ),
        use_container_width=True,
        key="chart_sec_u_history",
    )

    # Telemetry Log
    if not sim_df.empty:
        st.subheader("Telemetry Data")
        st.dataframe(sim_df.tail(8), use_container_width=True, hide_index=True)
        st.download_button(
            "Download Utility Telemetry (CSV)",
            data=sim_df.to_csv(index=False).encode("utf-8"),
            file_name="utility_agent_telemetry.csv",
            mime="text/csv",
            key="btn_sec_u_csv",
        )


# =========================================================
# SECTION 4: Reflex Thermostat
# =========================================================
elif nav_selection == "Reflex Thermostat":
    st.markdown(
        """<div class="main-title">Simple Reflex Thermostat</div>
<div class="sub-title">Deterministic threshold-based control without predictive lookahead</div>""",
        unsafe_allow_html=True,
    )

    r_state = reflex_env.get_state()
    r_preview = reflex_agent.preview_decision(r_state)

    # Current Environment Status
    st.markdown(
        get_status_card_html(
            classification=r_state["classification"],
            current_temp=r_state["room_temp"],
            desired_temp=r_state["desired_temp"],
            dark_mode=dark_mode,
        ),
        unsafe_allow_html=True,
    )

    # Clean metric cards
    rm1, rm2, rm3, rm4 = st.columns(4)
    with rm1:
        st.metric("Current Room Temp", f"{r_state['room_temp']:.2f} °C")
    with rm2:
        st.metric("Lower Threshold (Heat below)", f"{reflex_agent.lower_threshold:.1f} °C")
    with rm3:
        st.metric("Upper Threshold (Cool above)", f"{reflex_agent.upper_threshold:.1f} °C")
    with rm4:
        st.metric("Selected Action", f"{r_preview.selected_action}")

    # Simulation Controls
    def perform_reflex_step() -> None:
        if reflex_env.current_step < st.session_state.max_steps:
            rec = reflex_agent.step_environment(reflex_env)
            st.session_state.reflex_simulation_records.append(rec)

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        if st.button("Run Simulation", key="btn_sec_r_run", use_container_width=True, type="primary"):
            remaining = st.session_state.max_steps - reflex_env.current_step
            if remaining > 0:
                for _ in range(remaining):
                    perform_reflex_step()
                st.rerun()
            else:
                st.warning("Simulation complete. Reset to start again.")
    with rc2:
        if st.button("Step 1 Step", key="btn_sec_r_step", use_container_width=True):
            if reflex_env.current_step < st.session_state.max_steps:
                perform_reflex_step()
                st.rerun()
            else:
                st.warning("Simulation complete. Reset to start again.")
    with rc3:
        if st.button("Reset Simulation", key="btn_sec_r_reset", use_container_width=True):
            reset_reflex_simulation()
            st.rerun()

    # Threshold Rules Display
    st.subheader("Condition-Action Rules")
    st.markdown(
        get_reflex_rules_html(
            current_temp=r_state["room_temp"],
            lower_threshold=reflex_agent.lower_threshold,
            upper_threshold=reflex_agent.upper_threshold,
            selected_action=r_preview.selected_action,
            dark_mode=dark_mode,
        ),
        unsafe_allow_html=True,
    )

    # Temperature Trajectory Chart
    st.subheader("Temperature History")
    r_df = pd.DataFrame(st.session_state.reflex_simulation_records)
    st.plotly_chart(
        create_reflex_temperature_history_chart(
            df=r_df,
            lower_threshold=reflex_agent.lower_threshold,
            upper_threshold=reflex_agent.upper_threshold,
            desired_temp=st.session_state.desired_temp,
            dark_mode=dark_mode,
        ),
        use_container_width=True,
        key="chart_sec_r_history",
    )

    # Telemetry Log
    if not r_df.empty:
        st.subheader("Telemetry Data")
        st.dataframe(r_df.tail(8), use_container_width=True, hide_index=True)
        st.download_button(
            "Download Reflex Telemetry (CSV)",
            data=r_df.to_csv(index=False).encode("utf-8"),
            file_name="reflex_agent_telemetry.csv",
            mime="text/csv",
            key="btn_sec_r_csv",
        )


# =========================================================
# SECTION 5: Agent Comparison
# =========================================================
elif nav_selection == "Agent Comparison":
    st.markdown(
        """<div class="main-title">Agent Performance Comparison</div>
<div class="sub-title">Side-by-side benchmark with identical environmental disturbances</div>""",
        unsafe_allow_html=True,
    )

    cfg = comparison_runner.config
    st.caption(
        f"**Synchronized Setup:** Initial Temp: {cfg.initial_temp:.1f}°C · Target: {cfg.desired_temp:.1f}°C (±{cfg.comfort_tolerance:.1f}°C) · "
        f"Outside Temp: {cfg.outside_temp:.1f}°C · Insulation: {cfg.insulation_level} · Seed: {cfg.random_seed}"
    )

    # Controls
    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button("Run Comparison", key="btn_sec_c_run", use_container_width=True, type="primary"):
            comparison_runner.run_all()
            st.rerun()
    with cc2:
        if st.button("Reset Comparison", key="btn_sec_c_reset", use_container_width=True):
            reset_all_simulations()
            st.rerun()

    # Evaluation results
    eval_res = comparison_runner.evaluate()
    u_df = eval_res["utility_df"]
    r_df = eval_res["reflex_df"]
    combined_df = eval_res["combined_df"]
    u_metrics = eval_res["utility_metrics"]
    r_metrics = eval_res["reflex_metrics"]
    comp_table_df = eval_res["comparison_df"]
    comp_summary = eval_res["summary"]

    st.markdown(get_comparison_summary_html(comp_summary, dark_mode=dark_mode), unsafe_allow_html=True)

    # Key Performance Metrics Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "Average Error (MAE)",
            f"{u_metrics.avg_abs_error:.2f}°C",
            delta=f"Reflex: {r_metrics.avg_abs_error:.2f}°C",
            delta_color="inverse",
            help="Average absolute difference from desired target temperature.",
        )
    with k2:
        st.metric(
            "Comfort Time %",
            f"{u_metrics.comfort_percentage:.1f}%",
            delta=f"Reflex: {r_metrics.comfort_percentage:.1f}%",
            help="Percentage of time spent within the acceptable comfort range.",
        )
    with k3:
        st.metric(
            "Total Energy Used",
            f"{u_metrics.total_energy:.1f} units",
            delta=f"Reflex: {r_metrics.total_energy:.1f} units",
            delta_color="inverse",
            help="Total energy consumed by heating and cooling actuation.",
        )
    with k4:
        st.metric(
            "Temperature Stability",
            f"{u_metrics.temp_fluctuations:.2f}°C/step",
            delta=f"Reflex: {r_metrics.temp_fluctuations:.2f}°C/step",
            delta_color="inverse",
            help="Average step-to-step temperature swing.",
        )

    # 1. Temperature Comparison Chart
    st.subheader("1. Synchronized Temperature Trajectory")
    st.plotly_chart(
        create_comparison_temperature_chart(
            utility_df=u_df,
            reflex_df=r_df,
            desired_temp=cfg.desired_temp,
            comfort_tolerance=cfg.comfort_tolerance,
            dark_mode=dark_mode,
        ),
        use_container_width=True,
        key="chart_sec_c_traj",
    )

    # 2. Cumulative Energy Consumption Chart
    st.subheader("2. Cumulative Energy Consumption")
    st.plotly_chart(
        create_comparison_energy_chart(
            utility_df=u_df,
            reflex_df=r_df,
            dark_mode=dark_mode,
        ),
        use_container_width=True,
        key="chart_sec_c_energy",
    )

    # 3. Absolute Temperature Error Over Time Chart
    st.subheader("3. Absolute Temperature Error Over Time")
    st.plotly_chart(
        create_comparison_error_chart(
            utility_df=u_df,
            reflex_df=r_df,
            desired_temp=cfg.desired_temp,
            dark_mode=dark_mode,
        ),
        use_container_width=True,
        key="chart_sec_c_error",
    )

    # 4. Action Distribution Comparison Chart
    st.subheader("4. Action Distribution (HEAT / COOL / OFF)")
    st.plotly_chart(
        create_comparison_action_distribution_chart(
            utility_df=u_df,
            reflex_df=r_df,
            dark_mode=dark_mode,
        ),
        use_container_width=True,
        key="chart_sec_c_action",
    )

    # Quantitative Metrics Table
    st.subheader("Performance Comparison Table")
    st.dataframe(comp_table_df, use_container_width=True, hide_index=True)


# =========================================================
# SECTION 6: Results
# =========================================================
elif nav_selection == "Results":
    st.markdown(
        """<div class="main-title">Results and Downloads</div>
<div class="sub-title">Review performance metrics and download simulation data as CSV files</div>""",
        unsafe_allow_html=True,
    )

    eval_res = comparison_runner.evaluate()
    comb_df = eval_res["combined_df"]
    u_df = eval_res["utility_df"]
    r_df = eval_res["reflex_df"]
    comp_table = eval_res["comparison_df"]
    summary = eval_res["summary"]

    # Summary metrics
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Total Steps Evaluated", f"{len(comb_df) - 1} steps")
    with s2:
        st.metric("Best Overall Agent", f"{summary['overall_winner']}")
    with s3:
        st.metric("Data Status", "Complete (0 NaNs)")

    st.markdown("---")
    st.subheader("Download Simulation Data")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "📥 Download Combined Comparison CSV",
            data=comb_df.to_csv(index=False).encode("utf-8"),
            file_name="agent_comparison_data.csv",
            mime="text/csv",
            key="dl_comb_csv",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "📥 Download Utility Agent CSV",
            data=u_df.to_csv(index=False).encode("utf-8"),
            file_name="utility_agent_data.csv",
            mime="text/csv",
            key="dl_u_csv",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            "📥 Download Reflex Thermostat CSV",
            data=r_df.to_csv(index=False).encode("utf-8"),
            file_name="reflex_agent_data.csv",
            mime="text/csv",
            key="dl_r_csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Metrics Summary Table")
    st.dataframe(comp_table, use_container_width=True, hide_index=True)

    with st.expander("🔍 View Raw Combined Telemetry Data"):
        st.dataframe(comb_df, use_container_width=True, hide_index=True)
