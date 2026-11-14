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

import time

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
    create_occupancy_heat_chart,
    get_comparison_summary_html,
    get_decision_card_html,
    get_device_status_panel_html,
    get_live_room_simulation_html,
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
        "people_present": bool(SimulationDefaults.DEFAULT_PEOPLE_PRESENT),
        "num_people": int(SimulationDefaults.DEFAULT_NUM_PEOPLE),
        "room_size": float(SimulationDefaults.DEFAULT_ROOM_SIZE),
        "heat_gain_per_person": float(SimulationDefaults.DEFAULT_HEAT_GAIN_PER_PERSON),
        "sim_playback_speed": "Normal (250 ms/step)",
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
    if "comparison_has_run" not in st.session_state:
        st.session_state.comparison_has_run = False
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = None


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
        people_present=bool(st.session_state.people_present),
        num_people=int(st.session_state.num_people) if st.session_state.people_present else 0,
        room_size=float(st.session_state.room_size),
        heat_gain_per_person=float(st.session_state.heat_gain_per_person),
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
        people_present=bool(st.session_state.people_present),
        num_people=int(st.session_state.num_people) if st.session_state.people_present else 0,
        room_size=float(st.session_state.room_size),
        heat_gain_per_person=float(st.session_state.heat_gain_per_person),
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
        "people_present": init_obs.get("people_present", st.session_state.people_present),
        "num_people": init_obs.get("num_people", st.session_state.num_people),
        "room_size": init_obs.get("room_size", st.session_state.room_size),
        "delta_occupancy": init_obs.get("delta_occupancy", 0.0),
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
        "people_present": init_obs.get("people_present", st.session_state.people_present),
        "num_people": init_obs.get("num_people", st.session_state.num_people),
        "room_size": init_obs.get("room_size", st.session_state.room_size),
        "delta_occupancy": init_obs.get("delta_occupancy", 0.0),
    }]


def reset_all_simulations() -> None:
    reset_utility_simulation()
    reset_reflex_simulation()
    st.session_state.comparison_runner = ComparisonRunner(_build_comparison_config())
    st.session_state.comparison_has_run = False
    st.session_state.comparison_results = None


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
    if hasattr(sc, "people_present"):
        st.session_state.people_present = sc.people_present
        st.session_state.num_people = sc.num_people
        st.session_state.room_size = sc.room_size
    reset_all_simulations()


init_session_state()

# Global handles
env: RoomEnvironment = st.session_state.env
agent: UtilityBasedAgent = st.session_state.agent
reflex_env: RoomEnvironment = st.session_state.reflex_env
reflex_agent: SimpleReflexAgent = st.session_state.reflex_agent
comparison_runner: ComparisonRunner = st.session_state.comparison_runner

# Synchronize comparison runner with latest configuration when no simulation is active
latest_cfg = _build_comparison_config()
if not st.session_state.comparison_has_run and comparison_runner.config != latest_cfg:
    comparison_runner.reset(latest_cfg)


def perform_agent_step() -> None:
    current_env: RoomEnvironment = st.session_state.env
    current_agent: UtilityBasedAgent = st.session_state.agent
    if current_env.current_step < st.session_state.max_steps:
        obs = current_env.get_state()
        dec = current_agent.decide_action(obs)
        next_st = current_env.step(dec.selected_action)
        st.session_state.simulation_records.append({
            "step": current_env.current_step,
            "room_temp": obs["room_temp"],
            "outside_temp": obs["outside_temp"],
            "desired_temp": obs["desired_temp"],
            "action": dec.selected_action,
            "predicted_temp": dec.predicted_temp,
            "actual_temp": next_st["room_temp"],
            "utility_score": dec.utility_score,
            "energy_consumed": dec.energy_consumed,
            "classification": next_st["classification"],
            "people_present": next_st.get("people_present", current_env.people_present),
            "num_people": next_st.get("num_people", current_env.num_people),
            "room_size": next_st.get("room_size", current_env.room_size),
            "delta_occupancy": next_st.get("delta_occupancy", current_env.delta_occupancy),
        })


def perform_reflex_step() -> None:
    current_reflex_env: RoomEnvironment = st.session_state.reflex_env
    current_reflex_agent: SimpleReflexAgent = st.session_state.reflex_agent
    if current_reflex_env.current_step < st.session_state.max_steps:
        rec = current_reflex_agent.step_environment(current_reflex_env)
        st.session_state.reflex_simulation_records.append(rec)


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
    dropdown_bg = "#172B24"
    dropdown_option_bg = "#172B24"
    dropdown_text_primary = "#E8F1E9"
    dropdown_text_secondary = "#B5C9BA"
    dropdown_selected_bg = "#28543F"
    dropdown_selected_text = "#FFFFFF"
    dropdown_hover_bg = "#244A38"
    dropdown_border = "#456B55"
    dropdown_focus = "#72B58A"
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
    dropdown_bg = "#FFFFFF"
    dropdown_option_bg = "#FFFFFF"
    dropdown_text_primary = "#243B30"
    dropdown_text_secondary = "#5F7162"
    dropdown_selected_bg = "#E4F0E5"
    dropdown_selected_text = "#243B30"
    dropdown_hover_bg = "#EDF5ED"
    dropdown_border = "#D6E4D6"
    dropdown_focus = "#245C43"

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
header[data-testid="stHeader"] button[data-testid="baseButton-headerNoPadding"] {{
    margin-top: 17px !important;
    margin-left: 14px !important;
    z-index: 1000000 !important;
    color: {header_text} !important;
}}

/* Hide collapse button inside sidebar content (removes top-right << in sidebar) */
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] button[data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] button[kind="header"] {{
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
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

/* 5. Sidebar layout starts cleanly below fixed header without excessive top space */
section[data-testid="stSidebar"] {{
    top: 72px !important;
    height: calc(100vh - 72px) !important;
    max-height: calc(100vh - 72px) !important;
    border-right: 1px solid {sidebar_border} !important;
    background-color: {sidebar_bg} !important;
    z-index: 999970 !important;
    transition: background-color 0.2s ease, border-color 0.2s ease;
    overflow: hidden !important;
}}

/* 1. Eliminate Excessive Top Blank Space: hide default empty sidebar header */
div[data-testid="stSidebarHeader"],
div[data-testid="stSidebarHeader"] > div {{
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
}}

/* Sidebar Content Container: 18px horizontal padding, 8px top padding, independent smooth scrolling */
div[data-testid="stSidebarContent"] {{
    padding-top: 8px !important;
    padding-bottom: 14px !important;
    padding-left: 18px !important;
    padding-right: 18px !important;
    height: 100% !important;
    max-height: 100% !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
}}

/* Clean, subtle scrollbar for sidebar */
div[data-testid="stSidebarContent"]::-webkit-scrollbar {{
    width: 5px !important;
}}

div[data-testid="stSidebarContent"]::-webkit-scrollbar-track {{
    background: transparent !important;
}}

div[data-testid="stSidebarContent"]::-webkit-scrollbar-thumb {{
    background: {"#304438" if dark_mode else "#DCE5D8"} !important;
    border-radius: 4px !important;
}}

div[data-testid="stSidebarContent"]::-webkit-scrollbar-thumb:hover {{
    background: {"#5A9E6F" if dark_mode else "#829B78"} !important;
}}

div[data-testid="stSidebarUserContent"] {{
    padding: 0 !important;
    margin: 0 !important;
}}

section[data-testid="stSidebar"] * {{
    color: {text_primary} !important;
}}

/* 6. Biophilic Modern Sidebar Navigation - Filled Full-Width Buttons & No Circles */
/* Force outer Streamlit element containers to expand to 100% full width */
.st-key-nav_selection,
div[data-testid="stElementContainer"]:has(div[data-testid="stRadio"]),
div[data-testid="stElementContainer"]:has(.st-key-nav_selection),
section[data-testid="stSidebar"] div.stRadio,
section[data-testid="stSidebar"] div[data-testid="stRadio"],
div[data-testid="stRadio"],
div[data-testid="stRadio"] > div,
div[data-testid="stRadio"] div[data-testid="stRadioGroup"],
div[data-testid="stRadio"] [role="radiogroup"],
div[data-testid="stRadio"] [class*="ex50key0"] {{
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    gap: 5px !important; /* Compact vertical gap */
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    box-sizing: border-box !important;
}}

/* Entire radio option wrapper spanning 100% */
div[data-testid="stRadio"] [class*="ex50key1"],
div[data-testid="stRadio"] .react-aria-RadioField,
div[data-testid="stRadio"] [role="radiogroup"] > div,
div[data-testid="stRadioGroup"] > div {{
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}}

/* Completely hide all radio button circles (outer circle and inner dot) across all Streamlit & React-Aria versions */
div[data-testid="stRadio"] [class*="ex50key4"],
div[data-testid="stRadio"] [class*="ex50key5"],
[data-testid="stRadio"] [class*="ex50key4"],
[data-testid="stRadio"] [class*="ex50key5"],
[class*="ex50key4"],
[class*="ex50key5"],
[data-testid="stRadioOption"] [class*="ex50key4"],
[data-testid="stRadioOption"] [class*="ex50key5"],
div[data-testid="stRadio"] [data-testid="stRadioOption"] > div > div:first-child,
div[data-testid="stRadio"] [data-testid="stRadioOption"] > div > span:first-child,
[data-testid="stRadioOption"] > div > div:first-child,
[data-testid="stRadioOption"] > div > span:first-child,
[data-testid="stRadioOption"] input,
div[data-testid="stRadio"] input[type="radio"],
div[data-testid="stRadio"] .react-aria-RadioButton,
div[data-testid="stRadio"] [role="radio"] > div:first-child {{
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    min-width: 0 !important;
    min-height: 0 !important;
    max-width: 0 !important;
    max-height: 0 !important;
    opacity: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    pointer-events: none !important;
    position: absolute !important;
}}

/* Zero out any residual flex gap from the hidden circle */
[data-testid="stRadioOption"] > div,
[data-testid="stRadioOption"] [class*="ex50key3"],
div[data-testid="stRadio"] label > div,
div[data-testid="stRadio"] [data-testid="stRadioOption"] > div {{
    gap: 0 !important;
    width: 100% !important;
    min-width: 100% !important;
    display: flex !important;
    align-items: center !important;
}}

/* Filled navigation buttons spanning 100% full width (Default Inactive State) */
div[data-testid="stRadio"] [data-testid="stRadioOption"],
div[data-testid="stRadio"] [class*="ex50key2"],
div[data-testid="stRadio"] [role="radiogroup"] label,
div[data-testid="stRadio"] .react-aria-RadioField label,
[data-testid="stRadioOption"] {{
    height: 40px !important;
    min-height: 40px !important;
    max-height: 40px !important;
    background-color: {"#1A2821" if dark_mode else "#FFFFFF"} !important;
    border: 1px solid {border_color} !important;
    border-left: 3.5px solid transparent !important;
    border-radius: 8px !important;
    padding: 0 12px !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, {"0.15" if dark_mode else "0.03"}) !important;
}}

/* Inactive navigation hover */
div[data-testid="stRadio"] [data-testid="stRadioOption"]:not([data-selected]):not(:has(input:checked)):hover,
div[data-testid="stRadio"] [class*="ex50key2"]:not([data-selected]):not(:has(input:checked)):hover,
div[data-testid="stRadio"] [role="radiogroup"] label:not([data-selected]):not(:has(input:checked)):hover,
[data-testid="stRadioOption"]:not([data-selected]):not(:has(input:checked)):hover {{
    background-color: {"#24382D" if dark_mode else "#EAF2EB"} !important;
    border-color: {"#456E52" if dark_mode else "#B4D5BE"} !important;
    border-left-color: {"#5A9E6F" if dark_mode else "#829B78"} !important;
    transform: translateX(1px) !important;
}}

/* Active navigation state: prominent filled sage/forest pill spanning 100% width */
div[data-testid="stRadio"] [class*="ex50key1"][data-selected] [data-testid="stRadioOption"],
div[data-testid="stRadio"] [class*="ex50key1"][data-selected="true"] [data-testid="stRadioOption"],
div[data-testid="stRadio"] [class*="ex50key1"][data-selected] [class*="ex50key2"],
div[data-testid="stRadio"] [class*="ex50key1"][data-selected="true"] [class*="ex50key2"],
div[data-testid="stRadio"] .react-aria-RadioField[data-selected] [data-testid="stRadioOption"],
div[data-testid="stRadio"] .react-aria-RadioField[data-selected="true"] [data-testid="stRadioOption"],
div[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected],
div[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"],
div[data-testid="stRadio"] [data-testid="stRadioOption"]:has(input:checked),
div[data-testid="stRadio"] [class*="ex50key2"][data-selected],
div[data-testid="stRadio"] [class*="ex50key2"][data-selected="true"],
div[data-testid="stRadio"] [class*="ex50key2"]:has(input:checked),
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked),
div[data-testid="stRadio"] [role="radiogroup"] label[data-selected],
div[data-testid="stRadio"] [role="radiogroup"] label[data-selected="true"],
[data-testid="stRadioOption"][data-selected],
[data-testid="stRadioOption"][data-selected="true"],
[data-testid="stRadioOption"]:has(input:checked) {{
    background-color: {"#244030" if dark_mode else "#E2ECE3"} !important;
    border: 1px solid {"#3A6348" if dark_mode else "#B0CFB6"} !important;
    border-left: 3.5px solid {primary_green} !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, {"0.2" if dark_mode else "0.05"}) !important;
}}

/* Navigation label typography */
div[data-testid="stRadio"] [data-testid="stMarkdownContainer"],
div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p,
[data-testid="stRadioOption"] [data-testid="stMarkdownContainer"],
[data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p {{
    display: flex !important;
    align-items: center !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: {"#B4C4B5" if dark_mode else "#26382D"} !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
    white-space: nowrap !important;
    width: 100% !important;
}}

/* Icons inside navigation items: 19px outline icons */
div[data-testid="stRadio"] [data-testid="stIconMaterial"],
div[data-testid="stRadio"] .material-symbols-rounded,
[data-testid="stRadioOption"] [data-testid="stIconMaterial"],
[data-testid="stRadioOption"] .material-symbols-rounded {{
    color: {"#8A9E8E" if dark_mode else "#5F7162"} !important;
    font-size: 19px !important;
    margin-right: 10px !important;
    vertical-align: middle !important;
    transition: color 0.15s ease !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 20px !important;
    min-width: 20px !important;
}}

/* Active navigation text and icon */
div[data-testid="stRadio"] [class*="ex50key1"][data-selected] [data-testid="stMarkdownContainer"] p,
div[data-testid="stRadio"] [class*="ex50key1"][data-selected="true"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] p,
div[data-testid="stRadio"] [role="radiogroup"] label[data-selected="true"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stRadio"] [role="radiogroup"] label[data-selected] [data-testid="stMarkdownContainer"] p,
[data-testid="stRadioOption"][data-selected] [data-testid="stMarkdownContainer"] p,
[data-testid="stRadioOption"][data-selected="true"] [data-testid="stMarkdownContainer"] p,
[data-testid="stRadioOption"]:has(input:checked) [data-testid="stMarkdownContainer"] p {{
    color: {"#8FD0A8" if dark_mode else "#245C43"} !important;
    font-weight: 600 !important;
}}

div[data-testid="stRadio"] [class*="ex50key1"][data-selected] [data-testid="stIconMaterial"],
div[data-testid="stRadio"] [class*="ex50key1"][data-selected="true"] [data-testid="stIconMaterial"],
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) [data-testid="stIconMaterial"],
div[data-testid="stRadio"] [role="radiogroup"] label[data-selected="true"] [data-testid="stIconMaterial"],
div[data-testid="stRadio"] [role="radiogroup"] label[data-selected] [data-testid="stIconMaterial"],
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) .material-symbols-rounded,
div[data-testid="stRadio"] [role="radiogroup"] label[data-selected="true"] .material-symbols-rounded,
div[data-testid="stRadio"] [role="radiogroup"] label[data-selected] .material-symbols-rounded,
[data-testid="stRadioOption"][data-selected] [data-testid="stIconMaterial"],
[data-testid="stRadioOption"][data-selected="true"] [data-testid="stIconMaterial"],
[data-testid="stRadioOption"]:has(input:checked) [data-testid="stIconMaterial"],
[data-testid="stRadioOption"][data-selected] .material-symbols-rounded,
[data-testid="stRadioOption"][data-selected="true"] .material-symbols-rounded,
[data-testid="stRadioOption"]:has(input:checked) .material-symbols-rounded {{
    color: {"#8FD0A8" if dark_mode else "#245C43"} !important;
}}

/* Section Captions and Dividers - Compact */
.sidebar-section-caption {{
    font-size: 10.5px;
    font-weight: 700;
    color: {text_secondary};
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 0;
    margin-bottom: 6px;
    padding-left: 1px;
}}

.sidebar-divider {{
    height: 1px;
    background-color: {border_color};
    margin: 12px 0 8px 0; /* Compact spacing */
}}

.sidebar-settings-card {{
    background-color: {card_sub_bg};
    border: 1px solid {border_color};
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 12px;
    display: flex;
    flex-direction: column;
    gap: 5px;
}}

.sidebar-setting-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    padding: 0;
}}

.sidebar-setting-row .setting-k {{
    color: {text_secondary};
    font-weight: 400;
    font-size: 12px;
    text-align: left;
}}

.sidebar-setting-row .setting-v {{
    color: {text_primary};
    font-weight: 600;
    text-align: right;
    font-size: 12px;
    font-feature-settings: "tnum" 1;
}}

/* Sidebar Controls: Scenario Selectbox & Secondary Reset Button */
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [role="combobox"],
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-rac=""] {{
    min-height: 40px !important;
    border-radius: 8px !important;
    display: flex !important;
    align-items: center !important;
    background-color: {dropdown_bg} !important;
    border: 1.5px solid {dropdown_border} !important;
    color: {dropdown_text_primary} !important;
}}

section[data-testid="stSidebar"] button[kind="secondary"],
section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"],
section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {{
    height: 38px !important;
    min-height: 38px !important;
    border-radius: 8px !important;
    border: 1px solid {border_color} !important;
    background-color: {card_bg} !important;
    color: {text_primary} !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}}

section[data-testid="stSidebar"] button[kind="secondary"]:hover,
section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover {{
    background-color: {card_sub_bg} !important;
    border-color: {primary_green} !important;
    color: {primary_green} !important;
    transform: translateY(-1px) !important;
}}

/* 7. Selectbox & Dropdown Popovers (Full Theme Consistency & High Contrast) */
div[data-testid="stSelectbox"] label,
div[data-testid="stSelectbox"] label * {{
    color: {text_primary} !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
}}

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-testid="stSelectbox"] [role="combobox"],
div[data-baseweb="select"] > div {{
    background-color: {dropdown_bg} !important;
    background: {dropdown_bg} !important;
    border: 1.5px solid {dropdown_border} !important;
    border-radius: 8px !important;
    color: {dropdown_text_primary} !important;
    min-height: 40px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}}

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover,
div[data-testid="stSelectbox"] [role="combobox"]:hover {{
    border-color: {dropdown_focus} !important;
}}

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within,
div[data-testid="stSelectbox"] [role="combobox"]:focus,
div[data-testid="stSelectbox"] [role="combobox"]:focus-within {{
    border-color: {dropdown_focus} !important;
    box-shadow: 0 0 0 2px {dropdown_focus}50 !important;
    outline: none !important;
}}

div[data-testid="stSelectbox"] [data-baseweb="select"] span,
div[data-testid="stSelectbox"] [data-baseweb="select"] p,
div[data-testid="stSelectbox"] [data-baseweb="select"] div,
div[data-testid="stSelectbox"] [role="combobox"] span,
div[data-testid="stSelectbox"] [role="combobox"] p,
div[data-testid="stSelectbox"] [role="combobox"] div,
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [role="combobox"] * {{
    color: {dropdown_text_primary} !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    background: transparent !important;
    background-color: transparent !important;
}}

div[data-testid="stSelectbox"] svg,
div[data-baseweb="select"] svg,
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] svg {{
    fill: {dropdown_text_secondary} !important;
    color: {dropdown_text_secondary} !important;
}}

/* Selectbox Dropdown Menu Popover */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-testid="stSelectboxVirtualDropdown"],
ul[data-baseweb="menu"],
ul[role="listbox"],
div[role="listbox"] {{
    background-color: {dropdown_bg} !important;
    background: {dropdown_bg} !important;
    border: 1.5px solid {dropdown_border} !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 28px rgba(0, 0, 0, {"0.55" if dark_mode else "0.15"}) !important;
    z-index: 9999999 !important;
    max-height: 320px !important;
    overflow-y: auto !important;
    padding: 4px !important;
}}

/* Dropdown items / options */
li[data-baseweb="menu-item"],
li[role="option"],
ul[role="listbox"] > li,
div[role="listbox"] [role="option"],
div[data-testid="stSelectboxVirtualDropdown"] li {{
    background-color: {dropdown_option_bg} !important;
    background: {dropdown_option_bg} !important;
    color: {dropdown_text_primary} !important;
    font-size: 13.5px !important;
    font-weight: 450 !important;
    padding: 10px 14px !important;
    margin: 2px 0 !important;
    border-radius: 6px !important;
    min-height: 38px !important;
    height: auto !important;
    white-space: normal !important;
    word-break: break-word !important;
    line-height: 1.4 !important;
    cursor: pointer !important;
    transition: background-color 0.15s ease, color 0.15s ease !important;
    display: flex !important;
    align-items: center !important;
}}

/* Inner text of options */
li[data-baseweb="menu-item"] *,
li[role="option"] *,
ul[role="listbox"] > li *,
div[role="listbox"] [role="option"] *,
div[data-testid="stSelectboxVirtualDropdown"] li * {{
    color: {dropdown_text_primary} !important;
    background: transparent !important;
    background-color: transparent !important;
    white-space: normal !important;
    word-break: break-word !important;
}}

/* Hover state */
li[data-baseweb="menu-item"]:hover,
li[role="option"]:hover,
ul[role="listbox"] > li:hover,
div[role="listbox"] [role="option"]:hover,
div[data-testid="stSelectboxVirtualDropdown"] li:hover {{
    background-color: {dropdown_hover_bg} !important;
    background: {dropdown_hover_bg} !important;
    color: {dropdown_selected_text if dark_mode else dropdown_text_primary} !important;
}}

li[data-baseweb="menu-item"]:hover *,
li[role="option"]:hover *,
ul[role="listbox"] > li:hover *,
div[role="listbox"] [role="option"]:hover * {{
    color: {dropdown_selected_text if dark_mode else dropdown_text_primary} !important;
}}

/* Selected option state */
li[data-baseweb="menu-item"][aria-selected="true"],
li[role="option"][aria-selected="true"],
li[data-baseweb="menu-item"][aria-checked="true"],
li[role="option"][aria-checked="true"],
ul[role="listbox"] > li[aria-selected="true"],
div[role="listbox"] [role="option"][aria-selected="true"],
div[data-testid="stSelectboxVirtualDropdown"] li[aria-selected="true"] {{
    background-color: {dropdown_selected_bg} !important;
    background: {dropdown_selected_bg} !important;
    color: {dropdown_selected_text} !important;
    font-weight: 600 !important;
}}

li[data-baseweb="menu-item"][aria-selected="true"] *,
li[role="option"][aria-selected="true"] *,
ul[role="listbox"] > li[aria-selected="true"] *,
div[role="listbox"] [role="option"][aria-selected="true"] *,
div[data-testid="stSelectboxVirtualDropdown"] li[aria-selected="true"] * {{
    color: {dropdown_selected_text} !important;
    font-weight: 600 !important;
}}

/* Scrollbar for Dropdown Popover */
div[data-baseweb="popover"] ::-webkit-scrollbar,
ul[role="listbox"]::-webkit-scrollbar,
ul[data-baseweb="menu"]::-webkit-scrollbar {{
    width: 6px !important;
    height: 6px !important;
}}

div[data-baseweb="popover"] ::-webkit-scrollbar-track,
ul[role="listbox"]::-webkit-scrollbar-track {{
    background: {dropdown_bg} !important;
    border-radius: 4px !important;
}}

div[data-baseweb="popover"] ::-webkit-scrollbar-thumb,
ul[role="listbox"]::-webkit-scrollbar-thumb {{
    background: {dropdown_border} !important;
    border-radius: 4px !important;
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

/* =========================================================
   Live Room Environment Simulation & Device Panel Styles
   ========================================================= */

@keyframes fanSpinAnim {{
    from {{ transform: rotate(0deg); }}
    to {{ transform: rotate(360deg); }}
}}
@keyframes coldStreamAnim {{
    0% {{ stroke-dashoffset: 80; opacity: 0.15; }}
    50% {{ opacity: 0.95; }}
    100% {{ stroke-dashoffset: 0; opacity: 0.2; }}
}}
@keyframes snowflakeDriftAnim {{
    0% {{ transform: translate(0, 0) rotate(0deg); opacity: 0; }}
    25% {{ opacity: 0.95; }}
    80% {{ opacity: 0.85; }}
    100% {{ transform: translate(130px, 110px) rotate(160deg); opacity: 0; }}
}}
@keyframes heatWaveAnim {{
    0% {{ transform: translateY(0) scaleX(1); opacity: 0.15; }}
    40% {{ opacity: 0.9; }}
    80% {{ opacity: 0.7; }}
    100% {{ transform: translateY(-38px) scaleX(1.15); opacity: 0; }}
}}
@keyframes curtainFlutterAnim {{
    0% {{ transform: skewX(0deg) scaleX(1); }}
    50% {{ transform: skewX(3.5deg) scaleX(1.03); }}
    100% {{ transform: skewX(0deg) scaleX(1); }}
}}
@keyframes pulseDotAnim {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.5; transform: scale(1.2); }}
}}

.room-sim-container {{
    width: 100%;
    margin-bottom: 1.2rem;
    box-sizing: border-box;
}}

.room-sim-card {{
    background: {card_bg};
    border: 1px solid {border_color};
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, {"0.2" if dark_mode else "0.04"});
    box-sizing: border-box;
}}

.room-sim-topbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 12px;
    padding-bottom: 10px;
    border-bottom: 1px solid {border_color};
}}

.room-agent-badge {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 13.5px;
    font-weight: 700;
    color: {text_primary};
}}

.room-agent-badge .agent-pill {{
    background: {card_sub_bg};
    border: 1px solid {border_color};
    color: {forest_green};
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 12.5px;
    font-weight: 700;
}}

.room-sim-readouts {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}}

.room-pill {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 12.5px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid {border_color};
    background: {card_sub_bg};
    color: {text_primary};
}}

.room-pill.primary {{
    background: {"#244030" if dark_mode else "#E2ECE3"};
    border-color: {"#3A6348" if dark_mode else "#B0CFB6"};
    color: {"#8FD0A8" if dark_mode else "#245C43"};
}}

/* Mode Status Banner */
.room-status-banner {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 14px;
    border-radius: 8px;
    font-size: 13px;
    margin-bottom: 12px;
    border: 1px solid transparent;
}}

.room-status-banner.heating {{
    background: {"#2E1E12" if dark_mode else "#FDF1E6"};
    border-color: {"#5E3516" if dark_mode else "#F8CFA9"};
    color: {"#FFA258" if dark_mode else "#A64B15"};
}}

.room-status-banner.cooling {{
    background: {"#12222E" if dark_mode else "#EBF5FB"};
    border-color: {"#1B435E" if dark_mode else "#BCE0F5"};
    color: {"#68BFF5" if dark_mode else "#1B6291"};
}}

.room-status-banner.idle {{
    background: {card_sub_bg};
    border-color: {border_color};
    color: {text_secondary};
}}

.status-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    animation: pulseDotAnim 1.6s ease-in-out infinite;
}}

.status-dot.orange {{ background: #FF6B35; box-shadow: 0 0 8px #FF6B35; }}
.status-dot.blue {{ background: #00B4D8; box-shadow: 0 0 8px #00B4D8; }}
.status-dot.grey {{ background: #94A3B8; animation: none; }}

/* SVG Animation Classes */
.fan-spin {{
    transform-origin: 460px 85px;
    animation: fanSpinAnim 0.75s linear infinite;
}}

.cool-stream {{
    stroke-dasharray: 20 8;
    animation: coldStreamAnim 1.5s linear infinite;
}}
.cool-stream-2 {{ animation-delay: 0.3s; }}
.cool-stream-3 {{ animation-delay: 0.6s; }}

.snowflake {{
    animation: snowflakeDriftAnim 2.4s ease-in-out infinite;
}}
.sf-2 {{ animation-delay: 0.6s; }}
.sf-3 {{ animation-delay: 1.2s; }}
.sf-4 {{ animation-delay: 1.8s; }}

.heat-wave {{
    animation: heatWaveAnim 1.8s ease-out infinite;
}}
.hw-2 {{ animation-delay: 0.4s; }}
.hw-3 {{ animation-delay: 0.8s; }}

.curtain-flutter {{
    transform-origin: 540px 76px;
    animation: curtainFlutterAnim 2.8s ease-in-out infinite;
}}

@media (prefers-reduced-motion: reduce) {{
    .fan-spin, .cool-stream, .snowflake, .heat-wave, .curtain-flutter, .status-dot {{
        animation: none !important;
    }}
}}

/* Device Panel Grid */
.device-panel-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 1.4rem;
    box-sizing: border-box;
}}

.dev-card {{
    border-radius: 10px;
    padding: 14px 16px;
    transition: transform 0.15s ease, border-color 0.15s ease;
    box-sizing: border-box;
}}

.dev-card:hover {{
    transform: translateY(-1px);
}}

.dev-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}}

.dev-icon {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
}}

.dev-badge {{
    font-size: 11px;
    font-weight: 800;
    padding: 2.5px 8px;
    border-radius: 4px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}

.dev-name {{
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 2px;
    letter-spacing: -0.01em;
}}

.dev-subtext {{
    font-size: 12px;
    font-weight: 500;
}}

@media (max-width: 850px) {{
    .device-panel-grid {{
        grid-template-columns: repeat(2, 1fr);
    }}
}}

@media (max-width: 520px) {{
    .device-panel-grid {{
        grid-template-columns: 1fr;
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
    "Live Room Simulation",
    "Environment Setup",
    "Utility-Based Agent",
    "Reflex Thermostat",
    "Agent Comparison",
    "Results",
]

NAV_ICONS = {
    "Overview": ":material/dashboard:",
    "Live Room Simulation": ":material/view_in_ar:",
    "Environment Setup": ":material/tune:",
    "Utility-Based Agent": ":material/psychology:",
    "Reflex Thermostat": ":material/device_thermostat:",
    "Agent Comparison": ":material/balance:",
    "Results": ":material/bar_chart:",
}

def _on_sidebar_scenario_change() -> None:
    choice = st.session_state.get("sidebar_scenario_select", "")
    if "Scenario 1" in choice:
        apply_demonstration_scenario("cold_room")
    elif "Scenario 2" in choice:
        apply_demonstration_scenario("hot_room")
    elif "Scenario 3" in choice:
        apply_demonstration_scenario("comfortable_room")
    elif "Scenario 4" in choice:
        apply_demonstration_scenario("occupied_meeting")
    elif "Scenario 5" in choice:
        apply_demonstration_scenario("empty_office")
    elif "Custom" in choice:
        st.session_state.active_scenario_id = "custom"


with st.sidebar:
    nav_selection = st.radio(
        "Select Section",
        options=SECTIONS,
        format_func=lambda s: f"{NAV_ICONS.get(s, '')}  {s}",
        key="nav_selection",
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="sidebar-divider"></div><div class="sidebar-section-caption">DEMONSTRATION SCENARIOS</div>',
        unsafe_allow_html=True,
    )
    scenario_map = {
        "cold_room": 0,
        "hot_room": 1,
        "comfortable_room": 2,
        "occupied_meeting": 3,
        "empty_office": 4,
    }
    current_scenario_idx = scenario_map.get(st.session_state.active_scenario_id, 5)
    st.selectbox(
        "Load Preset Scenario",
        options=[
            "Scenario 1: Cold Room",
            "Scenario 2: Hot Room",
            "Scenario 3: Comfortable Room",
            "Scenario 4: Occupied Meeting (8 People)",
            "Scenario 5: Unoccupied Office (0 People)",
            "Custom Configuration",
        ],
        index=current_scenario_idx,
        key="sidebar_scenario_select",
        on_change=_on_sidebar_scenario_change,
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="sidebar-divider"></div><div class="sidebar-section-caption">PLAYBACK SPEED</div>',
        unsafe_allow_html=True,
    )
    speed_options = ["Normal (250 ms/step)", "Slow (500 ms/step)", "Fast (100 ms/step)"]
    current_speed = st.session_state.get("sim_playback_speed", "Normal (250 ms/step)")
    speed_idx = speed_options.index(current_speed) if current_speed in speed_options else 0
    selected_speed = st.selectbox(
        "Simulation Playback Speed",
        options=speed_options,
        index=speed_idx,
        key="sim_playback_speed",
        label_visibility="collapsed",
        help="Timestep delay during real-time progressive simulation playback.",
    )
    speed_delays = {
        "Slow (500 ms/step)": 0.50,
        "Normal (250 ms/step)": 0.25,
        "Fast (100 ms/step)": 0.10,
    }
    sim_delay = speed_delays.get(selected_speed, 0.25)

    occ_summary = f"{st.session_state.num_people} People" if st.session_state.people_present else "Unoccupied (0)"
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
<span class="setting-k">Occupancy</span>
<span class="setting-v">{occ_summary}</span>
</div>
<div class="sidebar-setting-row">
<span class="setting-k">Room Area</span>
<span class="setting-v">{st.session_state.room_size:.0f} m²</span>
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

    # 2.2 Occupancy & Thermal Mass Secondary Metrics
    u_preview = agent.decide_action(env.get_state())
    om1, om2, om3, om4 = st.columns(4)
    with om1:
        occ_str = f"{st.session_state.num_people} Occupants" if st.session_state.people_present else "Unoccupied"
        st.markdown(
            get_overview_metric_card_html(
                label="Room Occupancy",
                value=occ_str,
                subtext="Sensible metabolic body heat active" if st.session_state.people_present else "Zero occupant heat (0.00°C)",
                status_type="comfortable" if st.session_state.people_present else None,
                dark_mode=dark_mode,
            ),
            unsafe_allow_html=True,
        )
    with om2:
        st.markdown(
            get_overview_metric_card_html(
                label="Room Floor Area",
                value=f"{st.session_state.room_size:.0f} m²",
                subtext=f"Thermal scale: {25.0 / st.session_state.room_size:.2f}x (baseline: 25 m²)",
                status_type=None,
                dark_mode=dark_mode,
            ),
            unsafe_allow_html=True,
        )
    with om3:
        st.markdown(
            get_overview_metric_card_html(
                label="Body Heat Rate",
                value=f"+{env.delta_occupancy:.2f} °C",
                subtext="Heat gain per simulation step",
                status_type="hot" if env.delta_occupancy > 0 else None,
                dark_mode=dark_mode,
            ),
            unsafe_allow_html=True,
        )
    with om4:
        st.markdown(
            get_overview_metric_card_html(
                label="Predicted Next Temp",
                value=f"{u_preview.predicted_temp:.1f} °C",
                subtext=f"Forecast under {u_preview.selected_action}",
                status_type="comfortable" if env.classify_temperature(u_preview.predicted_temp) == CLASS_COMFORTABLE else ("cold" if u_preview.predicted_temp < st.session_state.desired_temp else "hot"),
                dark_mode=dark_mode,
            ),
            unsafe_allow_html=True,
        )

    # 2.5 Live Room Environment Simulation
    st.subheader("Live Room Environment Simulation")
    if len(st.session_state.simulation_records) > 1:
        ov_action = st.session_state.simulation_records[-1]["action"]
    else:
        ov_action = u_preview.selected_action
    
    ov_fan = (ov_action in [ACTION_HEAT, ACTION_COOL])
    ov_win = (ov_action == ACTION_OFF and 19.0 <= env.outside_temp <= 25.0)

    st.markdown(
        get_live_room_simulation_html(
            current_temp=cur_temp,
            desired_temp=st.session_state.desired_temp,
            outside_temp=st.session_state.outside_temp,
            classification=cur_class,
            action=ov_action,
            agent_name="Utility-Based Agent",
            fan_enabled=ov_fan,
            window_open=ov_win,
            comfort_tolerance=st.session_state.comfort_tolerance,
            dark_mode=dark_mode,
            people_present=st.session_state.people_present,
            num_people=st.session_state.num_people if st.session_state.people_present else 0,
            room_size=st.session_state.room_size,
            heat_gain_per_person=st.session_state.heat_gain_per_person,
            predicted_temp=u_preview.predicted_temp,
            delta_occupancy=env.delta_occupancy,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        get_device_status_panel_html(
            action=ov_action,
            fan_enabled=ov_fan,
            window_open=ov_win,
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
# SECTION 2: Live Room Environment Simulation
# =========================================================
elif nav_selection == "Live Room Simulation":
    st.markdown(
        """<div class="main-title">Live Room Environment Simulation</div>
<div class="sub-title">Real-Time Interactive Physical Simulation of Heating, Cooling, Airflow & Natural Ventilation</div>""",
        unsafe_allow_html=True,
    )

    # Agent Controller Switcher & Airflow Controls
    agent_sel_col, fan_col, win_col = st.columns([2, 1, 1])
    with agent_sel_col:
        active_sim_agent = st.radio(
            "Controlling Agent",
            options=["Utility-Based Agent", "Simple Reflex Thermostat"],
            horizontal=True,
            key="sim_active_agent_radio",
            help="Select which agent controls the room environment and thermal devices.",
        )
    with fan_col:
        fan_mode = st.selectbox(
            "Ceiling Fan Mode",
            options=["Auto (Circulate during HVAC)", "Always ON", "Always OFF"],
            index=0,
            key="sim_fan_mode_select",
            help="Controls convective ceiling fan circulation.",
        )
    with win_col:
        window_mode = st.selectbox(
            "Natural Ventilation",
            options=["Auto (Eco-ventilation)", "Always OPEN", "Always CLOSED"],
            index=0,
            key="sim_window_mode_select",
            help="Controls window opening for passive outdoor ventilation.",
        )

    # Derive state according to selected agent
    if active_sim_agent == "Utility-Based Agent":
        current_sim_env = env
        current_agent_obj = agent
        sim_step = env.current_step
        curr_obs = env.get_state()
        curr_temp = curr_obs["room_temp"]
        curr_desired = curr_obs["desired_temp"]
        curr_outside = curr_obs["outside_temp"]
        curr_class = curr_obs["classification"]
        if len(st.session_state.simulation_records) > 1:
            curr_action = st.session_state.simulation_records[-1]["action"]
        else:
            curr_action = agent.decide_action(curr_obs).selected_action
        hist_df = env.get_history_df()
        agent_expl = "Utility-Based Agent evaluates predicted outcomes for HEAT, COOL, and OFF using a thermodynamic simulation model, balancing user comfort against energy consumption and actuator stability."
    else:
        current_sim_env = reflex_env
        current_agent_obj = reflex_agent
        sim_step = reflex_env.current_step
        curr_obs = reflex_env.get_state()
        curr_temp = curr_obs["room_temp"]
        curr_desired = curr_obs["desired_temp"]
        curr_outside = curr_obs["outside_temp"]
        curr_class = curr_obs["classification"]
        if len(st.session_state.reflex_simulation_records) > 0:
            curr_action = st.session_state.reflex_simulation_records[-1]["action"]
        else:
            curr_action = reflex_agent.select_action(curr_temp)
        hist_df = reflex_env.get_history_df()
        agent_expl = f"Simple Reflex Thermostat operates purely on condition-action rules: HEAT when temp < {reflex_agent.lower_threshold:.1f}°C, COOL when temp > {reflex_agent.upper_threshold:.1f}°C, and OFF within the deadband."

    # Fan & Window states
    if fan_mode == "Auto (Circulate during HVAC)":
        fan_active = (curr_action in [ACTION_HEAT, ACTION_COOL])
    elif fan_mode == "Always ON":
        fan_active = True
    else:
        fan_active = False

    if window_mode == "Auto (Eco-ventilation)":
        window_active = (curr_action == ACTION_OFF and 19.0 <= curr_outside <= 25.0)
    elif window_mode == "Always OPEN":
        window_active = True
    else:
        window_active = False

    # 1. Main Visual Room Simulator
    pred_temp = agent.decide_action(curr_obs).predicted_temp if active_sim_agent == "Utility-Based Agent" else None
    sim_room_placeholder = st.empty()
    sim_room_placeholder.markdown(
        get_live_room_simulation_html(
            current_temp=curr_temp,
            desired_temp=curr_desired,
            outside_temp=curr_outside,
            classification=curr_class,
            action=curr_action,
            agent_name=active_sim_agent,
            fan_enabled=fan_active,
            window_open=window_active,
            comfort_tolerance=st.session_state.comfort_tolerance,
            dark_mode=dark_mode,
            people_present=st.session_state.people_present,
            num_people=st.session_state.num_people if st.session_state.people_present else 0,
            room_size=st.session_state.room_size,
            heat_gain_per_person=st.session_state.heat_gain_per_person,
            predicted_temp=pred_temp,
            delta_occupancy=current_sim_env.delta_occupancy,
        ),
        unsafe_allow_html=True,
    )

    # 2. Compact 4-Device Status Panel
    sim_device_placeholder = st.empty()
    sim_device_placeholder.markdown(
        get_device_status_panel_html(
            action=curr_action,
            fan_enabled=fan_active,
            window_open=window_active,
            dark_mode=dark_mode,
        ),
        unsafe_allow_html=True,
    )

    # 3. Interactive Simulation Controls & Visual Timeline
    controls_container = st.container()

    # 4. Live Synchronized Thermal Performance Chart
    st.subheader(f"Thermal State & Action Timeline ({active_sim_agent})")
    sim_chart_placeholder = st.empty()
    if len(hist_df) > 0:
        if active_sim_agent == "Utility-Based Agent":
            sim_chart_placeholder.plotly_chart(
                create_agent_temperature_history_chart(
                    df=hist_df,
                    desired_temp=curr_desired,
                    comfort_tolerance=st.session_state.comfort_tolerance,
                    dark_mode=dark_mode,
                ),
                use_container_width=True,
                key="sim_temp_chart_u",
            )
        else:
            sim_chart_placeholder.plotly_chart(
                create_reflex_temperature_history_chart(
                    df=hist_df,
                    lower_thresh=reflex_agent.lower_threshold,
                    upper_thresh=reflex_agent.upper_threshold,
                    desired_temp=curr_desired,
                    comfort_tolerance=st.session_state.comfort_tolerance,
                    dark_mode=dark_mode,
                ),
                use_container_width=True,
                key="sim_temp_chart_r",
            )

    with controls_container:
        st.subheader("Simulation Controls")
        sim_progress_placeholder = st.empty()
        ctl1, ctl2, ctl3 = st.columns(3)
        with ctl1:
            if st.button("▶ Run Full Simulation", key="btn_sim_run_full", type="primary", use_container_width=True):
                remaining = st.session_state.max_steps - current_sim_env.current_step
                if remaining > 0:
                    prog_bar = sim_progress_placeholder.progress(
                        0.0,
                        text=f"Starting simulation — Step {current_sim_env.current_step}/{st.session_state.max_steps} ({selected_speed})",
                    )
                    for step_idx in range(remaining):
                        if active_sim_agent == "Utility-Based Agent":
                            perform_agent_step()
                        else:
                            perform_reflex_step()

                        step_obs = current_sim_env.get_state()
                        step_temp = step_obs["room_temp"]
                        step_class = step_obs["classification"]
                        step_desired = step_obs["desired_temp"]
                        step_outside = step_obs["outside_temp"]

                        if active_sim_agent == "Utility-Based Agent":
                            step_action = agent.last_decision.selected_action if agent.last_decision else ACTION_OFF
                            step_pred = agent.last_decision.predicted_temp if agent.last_decision else None
                        else:
                            step_action = reflex_agent.last_action or ACTION_OFF
                            step_pred = None

                        if fan_mode == "Auto (Circulate during HVAC)":
                            step_fan = (step_action in [ACTION_HEAT, ACTION_COOL])
                        elif fan_mode == "Always ON":
                            step_fan = True
                        else:
                            step_fan = False

                        if window_mode == "Auto (Eco-ventilation)":
                            step_win = (step_action == ACTION_OFF and 19.0 <= step_outside <= 25.0)
                        elif window_mode == "Always OPEN":
                            step_win = True
                        else:
                            step_win = False

                        progress_ratio = (step_idx + 1) / remaining
                        prog_bar.progress(
                            progress_ratio,
                            text=f"Step {current_sim_env.current_step}/{st.session_state.max_steps} — Temp: {step_temp:.2f}°C ({step_class.upper()}) | Action: {step_action} | Target: {step_desired:.1f}°C",
                        )

                        sim_room_placeholder.markdown(
                            get_live_room_simulation_html(
                                current_temp=step_temp,
                                desired_temp=step_desired,
                                outside_temp=step_outside,
                                classification=step_class,
                                action=step_action,
                                agent_name=active_sim_agent,
                                fan_enabled=step_fan,
                                window_open=step_win,
                                comfort_tolerance=st.session_state.comfort_tolerance,
                                dark_mode=dark_mode,
                                people_present=st.session_state.people_present,
                                num_people=st.session_state.num_people if st.session_state.people_present else 0,
                                room_size=st.session_state.room_size,
                                heat_gain_per_person=st.session_state.heat_gain_per_person,
                                predicted_temp=step_pred,
                                delta_occupancy=current_sim_env.delta_occupancy,
                            ),
                            unsafe_allow_html=True,
                        )
                        sim_device_placeholder.markdown(
                            get_device_status_panel_html(
                                action=step_action,
                                fan_enabled=step_fan,
                                window_open=step_win,
                                dark_mode=dark_mode,
                            ),
                            unsafe_allow_html=True,
                        )
                        step_df = current_sim_env.get_history_df()
                        if len(step_df) > 0:
                            if active_sim_agent == "Utility-Based Agent":
                                sim_chart_placeholder.plotly_chart(
                                    create_agent_temperature_history_chart(
                                        df=step_df,
                                        desired_temp=step_desired,
                                        comfort_tolerance=st.session_state.comfort_tolerance,
                                        dark_mode=dark_mode,
                                    ),
                                    use_container_width=True,
                                    key=f"sim_live_chart_u_{current_sim_env.current_step}",
                                )
                            else:
                                sim_chart_placeholder.plotly_chart(
                                    create_reflex_temperature_history_chart(
                                        df=step_df,
                                        lower_thresh=reflex_agent.lower_threshold,
                                        upper_thresh=reflex_agent.upper_threshold,
                                        desired_temp=step_desired,
                                        comfort_tolerance=st.session_state.comfort_tolerance,
                                        dark_mode=dark_mode,
                                    ),
                                    use_container_width=True,
                                    key=f"sim_live_chart_r_{current_sim_env.current_step}",
                                )

                        time.sleep(sim_delay)

                    prog_bar.progress(1.0, text=f"Simulation Complete ({st.session_state.max_steps} steps) — Final Temp: {step_temp:.2f}°C")
                    st.rerun()
                else:
                    st.info("Simulation already complete for this agent. Reset to run again.")
        with ctl2:
            if st.button("⏭ Step Forward (+1 Step)", key="btn_sim_step_one", use_container_width=True):
                if current_sim_env.current_step < st.session_state.max_steps:
                    if active_sim_agent == "Utility-Based Agent":
                        perform_agent_step()
                    else:
                        perform_reflex_step()
                    st.rerun()
                else:
                    st.info("Simulation completed.")
        with ctl3:
            if st.button("🔄 Reset Agent Simulation", key="btn_sim_reset_active", use_container_width=True):
                if active_sim_agent == "Utility-Based Agent":
                    reset_utility_simulation()
                else:
                    reset_reflex_simulation()
                st.rerun()

        st.caption("Note: Advancing the simulation updates the physical thermodynamic environment and recalculates the agent's real-time action.")

    # 5. Controlling Agent Decision Insight Card
    st.markdown(
        f"""<div class="info-card">
<b>💡 Controlling Agent Insight ({active_sim_agent}):</b><br>
{agent_expl}<br>
<b>Current Step:</b> {sim_step} / {st.session_state.max_steps} · <b>Active Thermal Actuation:</b> <code>{curr_action}</code> · <b>Observed Room Temp:</b> {curr_temp:.2f}°C
</div>""",
        unsafe_allow_html=True,
    )


# =========================================================
# SECTION 3: Environment Setup
# =========================================================
elif nav_selection == "Environment Setup":
    st.markdown(
        """<div class="main-title">Environment Setup</div>
<div class="sub-title">Configure room temperatures, insulation parameters, and agent weights</div>""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌡️ Temperature Settings")
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

        st.markdown("#### 👥 Occupancy & Room Configuration")
        st.session_state.people_present = st.toggle(
            "People Present in Room",
            value=bool(st.session_state.people_present),
            key="tog_people_present",
            help="Toggle occupant presence. When OFF, internal body heat is 0.00°C and Utility Agent activates energy setback mode.",
        )
        
        # Disabled when room is unoccupied
        occ_disabled = not st.session_state.people_present
        current_num_people = int(st.session_state.num_people) if st.session_state.num_people >= 1 else 2
        st.session_state.num_people = st.slider(
            "Number of People (1–15)",
            min_value=int(SimulationDefaults.MIN_PEOPLE),
            max_value=int(SimulationDefaults.MAX_PEOPLE),
            value=current_num_people,
            step=1,
            disabled=occ_disabled,
            key="sl_num_people",
            help="Active only when People Present = Yes. Each occupant adds ~0.08°C metabolic heat at reference 25m².",
        )

        st.session_state.room_size = st.slider(
            "Room Floor Area (m²)",
            min_value=float(SimulationDefaults.MIN_ROOM_SIZE),
            max_value=float(SimulationDefaults.MAX_ROOM_SIZE),
            value=float(st.session_state.room_size),
            step=1.0,
            key="sl_room_size",
            help="Floor area in m². Larger rooms have greater thermal mass, slowing temperature changes from HVAC and body heat.",
        )

        # Dynamic physical feedback note
        eff_p = st.session_state.num_people if st.session_state.people_present else 0
        eff_occ_calc = (eff_p * float(st.session_state.heat_gain_per_person) * (25.0 / max(8.0, st.session_state.room_size)))
        thermal_scale_calc = 25.0 / max(8.0, st.session_state.room_size)
        st.info(
            f"💡 **Thermal Physics Summary:** Metabolic Heat Gain = **+{eff_occ_calc:.3f} °C/step** "
            f"({'Active' if st.session_state.people_present else 'Unoccupied (0.00°C)'}) · "
            f"Room Thermal Scale = **{thermal_scale_calc:.2f}x** (Baseline: 25 m², 2.5m ceiling height)."
        )

    with col2:
        st.markdown("#### 🏗️ Building & HVAC Settings")
        ins_opts = ["Poor", "Medium", "Good"]
        st.session_state.insulation_level = st.selectbox(
            "Room Insulation Level",
            options=ins_opts,
            index=ins_opts.index(st.session_state.insulation_level),
            key="sb_insulation",
            help="Determines the rate of heat exchange with outside air.",
        )
        st.session_state.heater_power = st.slider(
            "Heater Power (°C/step at 25m²)",
            min_value=0.5,
            max_value=3.0,
            value=float(st.session_state.heater_power),
            step=0.1,
            key="sl_heater_power",
            help="Maximum temperature rise delivered by heating unit at 25m².",
        )
        st.session_state.cooling_power = st.slider(
            "Cooling Power (°C/step at 25m²)",
            min_value=0.5,
            max_value=3.0,
            value=float(st.session_state.cooling_power),
            step=0.1,
            key="sl_cooling_power",
            help="Maximum temperature drop delivered by AC unit at 25m².",
        )
        st.session_state.disturbance_level = st.slider(
            "Disturbance Noise Level (σ)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.disturbance_level),
            step=0.05,
            key="sl_disturbance",
            help="Standard deviation of environmental thermal noise.",
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
            val = float(st.session_state.sl_reflex_lower)
            st.session_state.reflex_lower_slider = val
            if val >= float(st.session_state.reflex_upper_slider):
                new_upper = min(
                    float(ReflexThermostatDefaults.MAX_THRESHOLD),
                    round(val + 0.5, 1),
                )
                st.session_state.reflex_upper_slider = new_upper

        def _sync_reflex_upper() -> None:
            val = float(st.session_state.sl_reflex_upper)
            st.session_state.reflex_upper_slider = val
            if val <= float(st.session_state.reflex_lower_slider):
                new_lower = max(
                    float(ReflexThermostatDefaults.MIN_THRESHOLD),
                    round(val - 0.5, 1),
                )
                st.session_state.reflex_lower_slider = new_lower

        st.session_state.reflex_lower_slider = st.slider(
            "Lower Threshold (°C) — Heat below this",
            min_value=float(ReflexThermostatDefaults.MIN_THRESHOLD),
            max_value=float(ReflexThermostatDefaults.MAX_THRESHOLD - 0.5),
            value=float(st.session_state.reflex_lower_slider),
            step=0.5,
            key="sl_reflex_lower",
            on_change=_sync_reflex_lower,
        )
        st.session_state.reflex_upper_slider = st.slider(
            "Upper Threshold (°C) — Cool above this",
            min_value=float(ReflexThermostatDefaults.MIN_THRESHOLD + 0.5),
            max_value=float(ReflexThermostatDefaults.MAX_THRESHOLD),
            value=float(st.session_state.reflex_upper_slider),
            step=0.5,
            key="sl_reflex_upper",
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
    pred_val_str = f"{last_dec.predicted_temp:.2f} °C" if last_dec else f"{current_state['room_temp']:.2f} °C"
    occ_val_str = f"{st.session_state.num_people} People" if st.session_state.people_present else "Unoccupied"
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Current Room Temp", f"{current_state['room_temp']:.2f} °C")
    with m2:
        st.metric("Desired Target", f"{current_state['desired_temp']:.1f} °C")
    with m3:
        st.metric("Outside Temp", f"{current_state['outside_temp']:.1f} °C")
    with m4:
        st.metric("Room Area", f"{st.session_state.room_size:.0f} m²")
    with m5:
        st.metric("Occupancy", occ_val_str, delta=f"+{env.delta_occupancy:.2f}°C" if env.delta_occupancy > 0 else "0.00°C")
    with m6:
        st.metric("Predicted Next", pred_val_str)

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
                "people_present": next_st.get("people_present", env.people_present),
                "num_people": next_st.get("num_people", env.num_people),
                "room_size": next_st.get("room_size", env.room_size),
                "delta_occupancy": next_st.get("delta_occupancy", env.delta_occupancy),
            })

    # Simulation Controls & Live Visuals
    u_controls_container = st.container()

    # Visual Room Environment Simulator (Utility-Based Agent)
    st.subheader("Live Room Environment Simulation")
    u_preview = agent.decide_action(current_state)
    u_init_action = last_dec.selected_action if last_dec else u_preview.selected_action
    u_init_fan = (u_init_action in [ACTION_HEAT, ACTION_COOL])
    u_init_win = (u_init_action == ACTION_OFF and 19.0 <= current_state["outside_temp"] <= 25.0)
    u_room_placeholder = st.empty()
    u_room_placeholder.markdown(
        get_live_room_simulation_html(
            current_temp=current_state["room_temp"],
            desired_temp=current_state["desired_temp"],
            outside_temp=current_state["outside_temp"],
            classification=current_state["classification"],
            action=u_init_action,
            agent_name="Utility-Based Agent",
            fan_enabled=u_init_fan,
            window_open=u_init_win,
            comfort_tolerance=st.session_state.comfort_tolerance,
            dark_mode=dark_mode,
            people_present=st.session_state.people_present,
            num_people=st.session_state.num_people if st.session_state.people_present else 0,
            room_size=st.session_state.room_size,
            heat_gain_per_person=st.session_state.heat_gain_per_person,
            predicted_temp=last_dec.predicted_temp if last_dec else u_preview.predicted_temp,
            delta_occupancy=env.delta_occupancy,
        ),
        unsafe_allow_html=True,
    )
    u_dev_placeholder = st.empty()
    u_dev_placeholder.markdown(
        get_device_status_panel_html(
            action=u_init_action,
            fan_enabled=u_init_fan,
            window_open=u_init_win,
            dark_mode=dark_mode,
        ),
        unsafe_allow_html=True,
    )

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
    u_chart_placeholder = st.empty()
    u_chart_placeholder.plotly_chart(
        create_agent_temperature_history_chart(
            df=sim_df,
            desired_temp=st.session_state.desired_temp,
            comfort_tolerance=st.session_state.comfort_tolerance,
            dark_mode=dark_mode,
        ),
        use_container_width=True,
        key="chart_sec_u_history",
    )

    with u_controls_container:
        st.subheader("Simulation Controls")
        u_prog_placeholder = st.empty()
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Run Simulation", key="btn_u_run", use_container_width=True, type="primary"):
                remaining = st.session_state.max_steps - env.current_step
                if remaining > 0:
                    prog_bar = u_prog_placeholder.progress(
                        0.0,
                        text=f"Starting simulation — Step {env.current_step}/{st.session_state.max_steps} ({selected_speed})",
                    )
                    for step_idx in range(remaining):
                        perform_agent_step()

                        step_state = env.get_state()
                        step_dec = agent.last_decision
                        step_action = step_dec.selected_action if step_dec else ACTION_OFF
                        step_fan = (step_action in [ACTION_HEAT, ACTION_COOL])
                        step_win = (step_action == ACTION_OFF and 19.0 <= step_state["outside_temp"] <= 25.0)

                        progress_ratio = (step_idx + 1) / remaining
                        prog_bar.progress(
                            progress_ratio,
                            text=f"Step {env.current_step}/{st.session_state.max_steps} — Temp: {step_state['room_temp']:.2f}°C ({step_state['classification'].upper()}) | Action: {step_action} | Target: {step_state['desired_temp']:.1f}°C",
                        )

                        u_room_placeholder.markdown(
                            get_live_room_simulation_html(
                                current_temp=step_state["room_temp"],
                                desired_temp=step_state["desired_temp"],
                                outside_temp=step_state["outside_temp"],
                                classification=step_state["classification"],
                                action=step_action,
                                agent_name="Utility-Based Agent",
                                fan_enabled=step_fan,
                                window_open=step_win,
                                comfort_tolerance=st.session_state.comfort_tolerance,
                                dark_mode=dark_mode,
                                people_present=st.session_state.people_present,
                                num_people=st.session_state.num_people if st.session_state.people_present else 0,
                                room_size=st.session_state.room_size,
                                heat_gain_per_person=st.session_state.heat_gain_per_person,
                                predicted_temp=step_dec.predicted_temp if step_dec else None,
                                delta_occupancy=env.delta_occupancy,
                            ),
                            unsafe_allow_html=True,
                        )
                        u_dev_placeholder.markdown(
                            get_device_status_panel_html(
                                action=step_action,
                                fan_enabled=step_fan,
                                window_open=step_win,
                                dark_mode=dark_mode,
                            ),
                            unsafe_allow_html=True,
                        )
                        u_step_df = pd.DataFrame(st.session_state.simulation_records)
                        u_chart_placeholder.plotly_chart(
                            create_agent_temperature_history_chart(
                                df=u_step_df,
                                desired_temp=st.session_state.desired_temp,
                                comfort_tolerance=st.session_state.comfort_tolerance,
                                dark_mode=dark_mode,
                            ),
                            use_container_width=True,
                            key=f"live_chart_sec_u_{env.current_step}",
                        )
                        time.sleep(sim_delay)

                    prog_bar.progress(1.0, text=f"Simulation Complete ({st.session_state.max_steps} steps) — Final Temp: {step_state['room_temp']:.2f}°C")
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
    r_occ_str = f"{st.session_state.num_people} People" if st.session_state.people_present else "Unoccupied"
    rm1, rm2, rm3, rm4, rm5, rm6 = st.columns(6)
    with rm1:
        st.metric("Current Room Temp", f"{r_state['room_temp']:.2f} °C")
    with rm2:
        st.metric("Lower Threshold", f"{reflex_agent.lower_threshold:.1f} °C")
    with rm3:
        st.metric("Upper Threshold", f"{reflex_agent.upper_threshold:.1f} °C")
    with rm4:
        st.metric("Room Area", f"{st.session_state.room_size:.0f} m²")
    with rm5:
        st.metric("Occupancy", r_occ_str, delta=f"+{reflex_env.delta_occupancy:.2f}°C" if reflex_env.delta_occupancy > 0 else "0.00°C")
    with rm6:
        st.metric("Selected Action", f"{r_preview.selected_action}")

    # Simulation Controls
    def perform_reflex_step() -> None:
        if reflex_env.current_step < st.session_state.max_steps:
            rec = reflex_agent.step_environment(reflex_env)
            st.session_state.reflex_simulation_records.append(rec)

    # Simulation Controls & Live Visuals
    r_controls_container = st.container()

    # Visual Room Environment Simulator (Simple Reflex Thermostat)
    st.subheader("Live Room Environment Simulation")
    r_action = r_preview.selected_action
    r_fan = (r_action in [ACTION_HEAT, ACTION_COOL])
    r_win = (r_action == ACTION_OFF and 19.0 <= r_state["outside_temp"] <= 25.0)
    r_room_placeholder = st.empty()
    r_room_placeholder.markdown(
        get_live_room_simulation_html(
            current_temp=r_state["room_temp"],
            desired_temp=r_state["desired_temp"],
            outside_temp=r_state["outside_temp"],
            classification=r_state["classification"],
            action=r_action,
            agent_name="Simple Reflex Thermostat",
            fan_enabled=r_fan,
            window_open=r_win,
            comfort_tolerance=st.session_state.comfort_tolerance,
            dark_mode=dark_mode,
            people_present=st.session_state.people_present,
            num_people=st.session_state.num_people if st.session_state.people_present else 0,
            room_size=st.session_state.room_size,
            heat_gain_per_person=st.session_state.heat_gain_per_person,
            predicted_temp=None,
            delta_occupancy=reflex_env.delta_occupancy,
        ),
        unsafe_allow_html=True,
    )
    r_dev_placeholder = st.empty()
    r_dev_placeholder.markdown(
        get_device_status_panel_html(
            action=r_action,
            fan_enabled=r_fan,
            window_open=r_win,
            dark_mode=dark_mode,
        ),
        unsafe_allow_html=True,
    )

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
    r_chart_placeholder = st.empty()
    r_chart_placeholder.plotly_chart(
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

    with r_controls_container:
        st.subheader("Simulation Controls")
        r_prog_placeholder = st.empty()
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            if st.button("Run Simulation", key="btn_sec_r_run", use_container_width=True, type="primary"):
                remaining = st.session_state.max_steps - reflex_env.current_step
                if remaining > 0:
                    prog_bar = r_prog_placeholder.progress(
                        0.0,
                        text=f"Starting simulation — Step {reflex_env.current_step}/{st.session_state.max_steps} ({selected_speed})",
                    )
                    for step_idx in range(remaining):
                        perform_reflex_step()

                        step_state = reflex_env.get_state()
                        step_action = reflex_agent.last_action or ACTION_OFF
                        step_fan = (step_action in [ACTION_HEAT, ACTION_COOL])
                        step_win = (step_action == ACTION_OFF and 19.0 <= step_state["outside_temp"] <= 25.0)

                        progress_ratio = (step_idx + 1) / remaining
                        prog_bar.progress(
                            progress_ratio,
                            text=f"Step {reflex_env.current_step}/{st.session_state.max_steps} — Temp: {step_state['room_temp']:.2f}°C ({step_state['classification'].upper()}) | Action: {step_action} | Target: {step_state['desired_temp']:.1f}°C",
                        )

                        r_room_placeholder.markdown(
                            get_live_room_simulation_html(
                                current_temp=step_state["room_temp"],
                                desired_temp=step_state["desired_temp"],
                                outside_temp=step_state["outside_temp"],
                                classification=step_state["classification"],
                                action=step_action,
                                agent_name="Simple Reflex Thermostat",
                                fan_enabled=step_fan,
                                window_open=step_win,
                                comfort_tolerance=st.session_state.comfort_tolerance,
                                dark_mode=dark_mode,
                                people_present=st.session_state.people_present,
                                num_people=st.session_state.num_people if st.session_state.people_present else 0,
                                room_size=st.session_state.room_size,
                                heat_gain_per_person=st.session_state.heat_gain_per_person,
                                predicted_temp=None,
                                delta_occupancy=reflex_env.delta_occupancy,
                            ),
                            unsafe_allow_html=True,
                        )
                        r_dev_placeholder.markdown(
                            get_device_status_panel_html(
                                action=step_action,
                                fan_enabled=step_fan,
                                window_open=step_win,
                                dark_mode=dark_mode,
                            ),
                            unsafe_allow_html=True,
                        )
                        r_step_df = pd.DataFrame(st.session_state.reflex_simulation_records)
                        r_chart_placeholder.plotly_chart(
                            create_reflex_temperature_history_chart(
                                df=r_step_df,
                                lower_threshold=reflex_agent.lower_threshold,
                                upper_threshold=reflex_agent.upper_threshold,
                                desired_temp=st.session_state.desired_temp,
                                dark_mode=dark_mode,
                            ),
                            use_container_width=True,
                            key=f"live_chart_sec_r_{reflex_env.current_step}",
                        )
                        time.sleep(sim_delay)

                    prog_bar.progress(1.0, text=f"Simulation Complete ({st.session_state.max_steps} steps) — Final Temp: {step_state['room_temp']:.2f}°C")
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
    eff_occ_comp = (cfg.num_people * cfg.heat_gain_per_person * 25.0 / max(8.0, cfg.room_size)) if cfg.people_present else 0.0
    occ_str = f"{cfg.num_people} Occupants (+{eff_occ_comp:.2f}°C/step)" if cfg.people_present else "Unoccupied (0.00°C/step)"
    st.caption(
        f"**Synchronized Setup:** Initial Temp: {cfg.initial_temp:.1f}°C · Target: {cfg.desired_temp:.1f}°C (±{cfg.comfort_tolerance:.1f}°C) · "
        f"Outside Temp: {cfg.outside_temp:.1f}°C · Room Size: {cfg.room_size:.0f} m² · Occupancy: {occ_str} · Insulation: {cfg.insulation_level} · Seed: {cfg.random_seed}"
    )

    # Synchronized Controls Container & Visual Progression
    comp_controls_container = st.container()

    # Quick Scenario Presets for Occupancy and Room Area Benchmarks
    with st.expander("⚡ Quick Benchmark Scenarios (Occupancy Levels & Room Sizes)", expanded=False):
        st.markdown("**1. Occupancy Scenarios:**")
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            if st.button("🚫 Empty Room (0 Occupants)", key="btn_pre_empty", use_container_width=True):
                st.session_state.people_present = False
                st.session_state.num_people = 0
                if "run_animated_comp_fn" in locals():
                    run_animated_comp_fn()
                else:
                    comparison_runner.reset(_build_comparison_config())
                    comparison_runner.run_all()
                    st.session_state.comparison_results = comparison_runner.evaluate()
                    st.session_state.comparison_has_run = True
                    st.rerun()
        with pcol2:
            if st.button("👥 1–2 Occupants (2 People)", key="btn_pre_two", use_container_width=True):
                st.session_state.people_present = True
                st.session_state.num_people = 2
                if "run_animated_comp_fn" in locals():
                    run_animated_comp_fn()
                else:
                    comparison_runner.reset(_build_comparison_config())
                    comparison_runner.run_all()
                    st.session_state.comparison_results = comparison_runner.evaluate()
                    st.session_state.comparison_has_run = True
                    st.rerun()
        with pcol3:
            if st.button("👨‍👩‍👧‍👦 Multiple Occupants (6 People)", key="btn_pre_multi", use_container_width=True):
                st.session_state.people_present = True
                st.session_state.num_people = 6
                if "run_animated_comp_fn" in locals():
                    run_animated_comp_fn()
                else:
                    comparison_runner.reset(_build_comparison_config())
                    comparison_runner.run_all()
                    st.session_state.comparison_results = comparison_runner.evaluate()
                    st.session_state.comparison_has_run = True
                    st.rerun()

        st.markdown("**2. Room Size Scenarios:**")
        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            if st.button("📦 Small Room (12 m²)", key="btn_pre_small", use_container_width=True):
                st.session_state.room_size = 12.0
                if "run_animated_comp_fn" in locals():
                    run_animated_comp_fn()
                else:
                    comparison_runner.reset(_build_comparison_config())
                    comparison_runner.run_all()
                    st.session_state.comparison_results = comparison_runner.evaluate()
                    st.session_state.comparison_has_run = True
                    st.rerun()
        with scol2:
            if st.button("🏠 Medium Room (25 m²)", key="btn_pre_med", use_container_width=True):
                st.session_state.room_size = 25.0
                if "run_animated_comp_fn" in locals():
                    run_animated_comp_fn()
                else:
                    comparison_runner.reset(_build_comparison_config())
                    comparison_runner.run_all()
                    st.session_state.comparison_results = comparison_runner.evaluate()
                    st.session_state.comparison_has_run = True
                    st.rerun()
        with scol3:
            if st.button("🏢 Large Room (60 m²)", key="btn_pre_large", use_container_width=True):
                st.session_state.room_size = 60.0
                if "run_animated_comp_fn" in locals():
                    run_animated_comp_fn()
                else:
                    comparison_runner.reset(_build_comparison_config())
                    comparison_runner.run_all()
                    st.session_state.comparison_results = comparison_runner.evaluate()
                    st.session_state.comparison_has_run = True
                    st.rerun()

    # Authoritative evaluation results (reads from st.session_state.comparison_results if completed)
    if st.session_state.comparison_has_run and st.session_state.comparison_results is not None:
        eval_res = st.session_state.comparison_results
    else:
        eval_res = comparison_runner.evaluate()

    u_df = eval_res["utility_df"]
    r_df = eval_res["reflex_df"]
    combined_df = eval_res["combined_df"]
    u_metrics = eval_res["utility_metrics"]
    r_metrics = eval_res["reflex_metrics"]
    comp_table_df = eval_res["comparison_df"]
    comp_summary = eval_res["summary"]
    has_run = bool(eval_res.get("has_run", st.session_state.comparison_has_run))

    st.markdown(get_comparison_summary_html(comp_summary, dark_mode=dark_mode), unsafe_allow_html=True)

    # Key Performance Metrics Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "Average Error (MAE)",
            f"{u_metrics.avg_abs_error:.2f}°C" if has_run else "N/A",
            delta=f"Reflex: {r_metrics.avg_abs_error:.2f}°C" if has_run else "Reflex: N/A",
            delta_color="inverse",
            help="Average absolute difference from desired target temperature.",
        )
    with k2:
        st.metric(
            "Comfort Time %",
            f"{u_metrics.comfort_percentage:.1f}%" if has_run else "N/A",
            delta=f"Reflex: {r_metrics.comfort_percentage:.1f}%" if has_run else "Reflex: N/A",
            help="Percentage of time spent within the acceptable comfort range.",
        )
    with k3:
        st.metric(
            "Total Energy Used",
            f"{u_metrics.total_energy:.1f} units" if has_run else "N/A",
            delta=f"Reflex: {r_metrics.total_energy:.1f} units" if has_run else "Reflex: N/A",
            delta_color="inverse",
            help="Total energy consumed by heating and cooling actuation.",
        )
    with k4:
        st.metric(
            "Temperature Stability",
            f"{u_metrics.temp_fluctuations:.2f}°C/step" if has_run else "N/A",
            delta=f"Reflex: {r_metrics.temp_fluctuations:.2f}°C/step" if has_run else "Reflex: N/A",
            delta_color="inverse",
            help="Average step-to-step temperature swing.",
        )

    def run_animated_comp_fn() -> None:
        fresh_cfg = _build_comparison_config()
        comparison_runner.reset(fresh_cfg)
        total_steps = comparison_runner.max_steps
        prog_bar = comp_prog_placeholder.progress(
            0.0,
            text=f"Starting synchronized comparison — 0/{total_steps} steps ({selected_speed})",
        )
        while not comparison_runner.is_finished:
            step_idx = comparison_runner.current_step
            if not comparison_runner.step_once():
                break
            u_df_live = comparison_runner.util_env.get_history_df()
            r_df_live = comparison_runner.refl_env.get_history_df()
            u_temp = u_df_live["actual_temp"].iloc[-1] if not u_df_live.empty else 0.0
            r_temp = r_df_live["actual_temp"].iloc[-1] if not r_df_live.empty else 0.0
            u_act = comparison_runner.util_agent.last_decision.selected_action if comparison_runner.util_agent.last_decision else ACTION_OFF
            r_act = comparison_runner.refl_agent.last_action or ACTION_OFF

            step_num = step_idx + 1
            prog_bar.progress(
                min(1.0, step_num / total_steps),
                text=f"Synchronized Step {step_num}/{total_steps} — Utility: {u_temp:.2f}°C [{u_act}] | Reflex: {r_temp:.2f}°C [{r_act}] | Target: {fresh_cfg.desired_temp:.1f}°C",
            )
            comp_chart_placeholder.plotly_chart(
                create_comparison_temperature_chart(
                    utility_df=u_df_live,
                    reflex_df=r_df_live,
                    desired_temp=fresh_cfg.desired_temp,
                    comfort_tolerance=fresh_cfg.comfort_tolerance,
                    dark_mode=dark_mode,
                ),
                use_container_width=True,
                key=f"comp_live_chart_{step_num}",
            )
            time.sleep(sim_delay)

        prog_bar.progress(1.0, text=f"Synchronized Comparison Complete ({total_steps} steps) — Computing benchmarks...")
        st.session_state.comparison_results = comparison_runner.evaluate()
        st.session_state.comparison_has_run = True
        st.rerun()

    with comp_controls_container:
        comp_prog_placeholder = st.empty()
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("Run Comparison", key="btn_sec_c_run", use_container_width=True, type="primary"):
                run_animated_comp_fn()
        with cc2:
            if st.button("Reset Comparison", key="btn_sec_c_reset", use_container_width=True):
                reset_all_simulations()
                st.session_state.comparison_has_run = False
                st.session_state.comparison_results = None
                st.rerun()

    # Comparison Trajectory Chart Placeholder
    comp_chart_placeholder = st.empty()
    if has_run:
        # Comparison Charts (clean visual presentation without redundant stacked headers)
        comp_chart_placeholder.plotly_chart(
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

        st.plotly_chart(
            create_comparison_energy_chart(
                utility_df=u_df,
                reflex_df=r_df,
                dark_mode=dark_mode,
            ),
            use_container_width=True,
            key="chart_sec_c_energy",
        )

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

        st.plotly_chart(
            create_comparison_action_distribution_chart(
                utility_df=u_df,
                reflex_df=r_df,
                dark_mode=dark_mode,
            ),
            use_container_width=True,
            key="chart_sec_c_action",
        )

        st.plotly_chart(
            create_occupancy_heat_chart(
                combined_df=combined_df,
                dark_mode=dark_mode,
            ),
            use_container_width=True,
            key="chart_sec_c_occ_heat",
        )
    else:
        st.info("💡 **Simulation Not Yet Executed** — Click **Run Comparison** above or select a preset benchmark scenario to generate comparative trajectory, energy, error, and occupancy heat charts.")

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

    if st.session_state.comparison_has_run and st.session_state.comparison_results is not None:
        eval_res = st.session_state.comparison_results
    else:
        eval_res = comparison_runner.evaluate()

    comb_df = eval_res["combined_df"]
    u_df = eval_res["utility_df"]
    r_df = eval_res["reflex_df"]
    comp_table = eval_res["comparison_df"]
    summary = eval_res["summary"]
    has_run = bool(eval_res.get("has_run", st.session_state.comparison_has_run))
    n_evaluated_steps = (len(comb_df) - 1) if has_run and not comb_df.empty else 0

    # Summary metrics
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Total Steps Evaluated", f"{n_evaluated_steps} steps")
    with s2:
        st.metric("Best Overall Agent", f"{summary['overall_winner']}")
    with s3:
        st.metric("Data Status", "Complete (0 NaNs)" if has_run else "Pending Execution")

    if not has_run:
        st.warning("⚠️ **No simulation results available yet.** Please run a simulation from the **Agent Comparison** page first to generate benchmark metrics and enable CSV downloads.")

    st.markdown("---")
    st.subheader("Download Simulation Data")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "📥 Download Combined Comparison CSV",
            data=comb_df.to_csv(index=False).encode("utf-8") if has_run else b"",
            file_name="agent_comparison_data.csv",
            mime="text/csv",
            key="dl_comb_csv",
            use_container_width=True,
            disabled=not has_run,
            help="Download side-by-side synchronized comparison telemetry across all steps." if has_run else "Run comparison first to enable download.",
        )
    with d2:
        st.download_button(
            "📥 Download Utility Agent CSV",
            data=u_df.to_csv(index=False).encode("utf-8") if has_run else b"",
            file_name="utility_agent_data.csv",
            mime="text/csv",
            key="dl_u_csv",
            use_container_width=True,
            disabled=not has_run,
            help="Download Utility-Based Agent decision log and metrics." if has_run else "Run comparison first to enable download.",
        )
    with d3:
        st.download_button(
            "📥 Download Reflex Thermostat CSV",
            data=r_df.to_csv(index=False).encode("utf-8") if has_run else b"",
            file_name="reflex_agent_data.csv",
            mime="text/csv",
            key="dl_r_csv",
            use_container_width=True,
            disabled=not has_run,
            help="Download Simple Reflex Thermostat step telemetry." if has_run else "Run comparison first to enable download.",
        )

    st.markdown("---")
    st.subheader("Metrics Summary Table")
    st.dataframe(comp_table, use_container_width=True, hide_index=True)

    with st.expander("🔍 View Raw Combined Telemetry Data"):
        if has_run and not comb_df.empty:
            telemetry_display_cols = {
                "step": "Step",
                "outside_temp": "Outside Temp (°C)",
                "desired_temp": "Target Temp (°C)",
                "lower_comfort": "Lower Comfort (°C)",
                "upper_comfort": "Upper Comfort (°C)",
                "people_present": "People Present",
                "num_people": "Occupant Count",
                "room_size": "Room Area (m²)",
                "delta_occupancy": "Body Heat Gain (°C)",
                "disturbance": "Disturbance (°C)",
                "utility_room_temp": "Utility Room Temp (°C)",
                "utility_action": "Utility Action",
                "utility_predicted_temp": "Utility Predicted Temp (°C)",
                "utility_actual_temp": "Utility Next Temp (°C)",
                "utility_abs_error": "Utility Abs Error (°C)",
                "utility_score": "Utility Score",
                "utility_energy_consumed": "Utility Energy Step",
                "utility_cumulative_energy": "Utility Energy Total",
                "utility_classification": "Utility Classification",
                "reflex_room_temp": "Reflex Room Temp (°C)",
                "reflex_action": "Reflex Action",
                "reflex_actual_temp": "Reflex Next Temp (°C)",
                "reflex_abs_error": "Reflex Abs Error (°C)",
                "reflex_energy_consumed": "Reflex Energy Step",
                "reflex_cumulative_energy": "Reflex Energy Total",
                "reflex_classification": "Reflex Classification",
            }
            display_df = comb_df.rename(columns=telemetry_display_cols)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No simulation telemetry recorded yet. Run a comparison from the Agent Comparison page to populate telemetry data.")
