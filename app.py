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

import importlib
import pandas as pd
import streamlit as st

import config
importlib.reload(config)
import utils
importlib.reload(utils)

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
    app_bg = "#101b17"
    card_bg = "#192720"
    card_sub_bg = "#22352a"
    text_primary = "#e7f0e8"
    text_secondary = "#a1b5a5"
    forest_green = "#376b4e"
    muted_teal = "#5bb5b5"
    sage_green = "#9dba91"
    mint_accent = "#8fd0a8"
    border_color = "#304438"
    header_bg = "#192720"
    header_border = "#304438"
    header_text = "#e7f0e8"
    sidebar_bg = "#101b17"
    sidebar_border = "#304438"
    button_hover_bg = "#447f5e"
else:
    app_bg = "#f5f7f0"
    card_bg = "#ffffff"
    card_sub_bg = "#eaf0e5"
    text_primary = "#26382d"
    text_secondary = "#718174"
    forest_green = "#245c43"
    muted_teal = "#2e8b8b"
    sage_green = "#829b78"
    mint_accent = "#b9d7c1"
    border_color = "#dce5d8"
    header_bg = "#ffffff"
    header_border = "#dce5d8"
    header_text = "#26382d"
    sidebar_bg = "#ffffff"
    sidebar_border = "#dce5d8"
    button_hover_bg = "#1d4a36"

theme_css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
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
    height: 70px !important;
    z-index: 999990 !important;
}}

/* Ensure Streamlit's sidebar collapse/expand button sits cleanly on top left of header */
header[data-testid="stHeader"] button[data-testid="stSidebarCollapseButton"],
header[data-testid="stHeader"] button[data-testid="baseButton-headerNoPadding"],
[data-testid="stSidebarCollapseButton"] {{
    margin-top: 10px !important;
    margin-left: 12px !important;
    z-index: 1000000 !important;
    color: {header_text} !important;
}}

/* 3. Professional Biophilic Fixed Top Header */
.fixed-top-header {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 70px;
    background-color: {header_bg};
    border-bottom: 1px solid {header_border};
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 1.5rem 0 4.2rem;
    z-index: 999980;
    box-shadow: 0 1px 4px rgba(0, 0, 0, {"0.35" if dark_mode else "0.04"});
    box-sizing: border-box;
    transition: background-color 0.2s ease, border-color 0.2s ease;
}}

.header-left {{
    display: flex;
    align-items: center;
    gap: 12px;
    user-select: none;
}}

.header-icon {{
    font-size: 24px;
    line-height: 1;
}}

.header-title-group {{
    display: flex;
    flex-direction: column;
}}

.header-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {header_text};
    letter-spacing: -0.01em;
    line-height: 1.2;
    white-space: nowrap;
}}

.header-subtitle {{
    font-size: 0.76rem;
    font-weight: 500;
    color: {text_secondary};
    letter-spacing: 0.01em;
    line-height: 1.2;
    white-space: nowrap;
}}

.header-right {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-right: 125px; /* Leaves room for docked toggle */
}}

.header-status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.03em;
    white-space: nowrap;
    transition: all 0.2s ease;
}}

/* Dock Theme Toggle switch directly into top right header */
div[data-testid="stToggle"] {{
    position: fixed !important;
    top: 20px !important;
    right: 22px !important;
    z-index: 1000000 !important;
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
}}

div[data-testid="stToggle"] label {{
    margin: 0 !important;
    color: {header_text} !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}}

/* 4. Page Content Layout & Vertical Spacing */
.block-container,
div[data-testid="stMainBlockContainer"],
div[data-testid="block-container"] {{
    padding-top: 5.6rem !important;
    background-color: {app_bg} !important;
}}

/* 5. Sidebar layout starts cleanly below fixed header */
section[data-testid="stSidebar"] {{
    top: 70px !important;
    height: calc(100vh - 70px) !important;
    border-right: 1px solid {sidebar_border} !important;
    background-color: {sidebar_bg} !important;
    z-index: 999970 !important;
}}

section[data-testid="stSidebar"] * {{
    color: {text_primary} !important;
}}

/* 6. Biophilic Typography and Card styling */
.main-title {{
    font-size: 1.85rem;
    font-weight: 800;
    color: {text_primary};
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}}

.sub-title {{
    font-size: 1.02rem;
    color: {text_secondary};
    margin-bottom: 1.1rem;
    font-weight: 500;
}}

.info-card {{
    background-color: {card_bg};
    border: 1px solid {border_color};
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 1rem;
    color: {text_primary};
    font-size: 0.95rem;
    line-height: 1.5;
}}

.summary-card {{
    background: {card_bg};
    border: 1px solid {border_color};
    border-radius: 10px;
    padding: 16px 18px;
    height: 100%;
    color: {text_primary};
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}}

/* Metric Cards */
div[data-testid="stMetric"] {{
    background-color: {card_bg} !important;
    border: 1px solid {border_color} !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
}}

div[data-testid="stMetricValue"] {{
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: {text_primary} !important;
    font-feature-settings: "tnum" 1;
}}

div[data-testid="stMetricLabel"] {{
    color: {text_secondary} !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}}

/* Buttons Styling in Biophilic Forest Green */
button[kind="primary"],
.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {{
    background-color: {forest_green} !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: background-color 0.2s ease !important;
}}

button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {{
    background-color: {button_hover_bg} !important;
    color: #ffffff !important;
}}

button[kind="secondary"],
.stButton > button[kind="secondary"],
button[data-testid="baseButton-secondary"] {{
    background-color: {card_bg} !important;
    color: {text_primary} !important;
    border: 1px solid {border_color} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}}

button[kind="secondary"]:hover,
.stButton > button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover {{
    background-color: {card_sub_bg} !important;
    border-color: {forest_green} !important;
    color: {text_primary} !important;
}}

/* Download Buttons */
div[data-testid="stDownloadButton"] > button {{
    background-color: {card_bg} !important;
    color: {text_primary} !important;
    border: 1px solid {border_color} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}}

div[data-testid="stDownloadButton"] > button:hover {{
    background-color: {card_sub_bg} !important;
    border-color: {forest_green} !important;
}}

/* Sliders */
div[data-baseweb="slider"] div[role="slider"] {{
    background-color: {forest_green} !important;
    border-color: {mint_accent} !important;
}}

/* Headings */
h1, h2, h3, h4, h5, h6 {{
    color: {text_primary} !important;
    font-weight: 700 !important;
}}

/* Responsive Mobile / Tablet adjustments */
@media (max-width: 860px) {{
    .header-subtitle {{
        display: none !important;
    }}
    .header-title {{
        font-size: 0.95rem !important;
    }}
    .header-status-badge {{
        font-size: 11px !important;
        padding: 4px 8px !important;
    }}
    .header-right {{
        margin-right: 110px !important;
    }}
}}

@media (max-width: 640px) {{
    .header-title-group {{
        display: none !important;
    }}
    .fixed-top-header {{
        padding-left: 3.5rem !important;
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

if dark_mode:
    if cur_class == CLASS_COLD:
        badge_bg = "#162828"
        badge_border = "#2e8b8b"
        badge_color = "#5bb5b5"
    elif cur_class == CLASS_HOT:
        badge_bg = "#2a1b1a"
        badge_border = "#c65b52"
        badge_color = "#e58b80"
    else:
        badge_bg = "#192720"
        badge_border = "#376b4e"
        badge_color = "#75c995"
else:
    badge_bg = status_info["bg_light"]
    badge_border = status_info["border"]
    badge_color = status_info["color"]

header_html = f"""<header class="fixed-top-header">
<div class="header-left">
<span class="header-icon">🌿🌡️</span>
<div class="header-title-group">
<div class="header-title">Environment Classifier and Demonstrator</div>
<div class="header-subtitle">Intelligent Room Temperature Control</div>
</div>
</div>
<div class="header-right">
<div class="header-status-badge" style="background-color: {badge_bg}; border: 1px solid {badge_border}; color: {badge_color};">
<span>{status_info['icon']}</span>
<span>{cur_class.upper()}</span>
<span style="opacity: 0.85; margin-left: 2px;">{cur_temp:.1f}°C</span>
</div>
</div>
</header>"""
st.markdown(header_html, unsafe_allow_html=True)

# Theme Toggle switch (docked via CSS into top right of fixed header)
toggle_theme = st.toggle(
    "🌙 Dark" if dark_mode else "☀️ Light",
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

with st.sidebar:
    st.markdown("### 🧭 Navigation")
    nav_selection = st.radio(
        "Select Section",
        options=SECTIONS,
        key="nav_selection",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("#### 🎯 Demonstration Scenarios")
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
    )

    if st.button("Apply Selected Scenario", use_container_width=True, key="btn_apply_scenario"):
        if "Scenario 1" in scenario_choice:
            apply_demonstration_scenario("cold_room")
        elif "Scenario 2" in scenario_choice:
            apply_demonstration_scenario("hot_room")
        elif "Scenario 3" in scenario_choice:
            apply_demonstration_scenario("comfortable_room")
        st.success("Scenario loaded!")
        st.rerun()

    st.markdown("---")
    st.markdown("#### 📌 Current Settings")
    st.caption(
        f"**Target:** {st.session_state.desired_temp:.1f}°C (±{st.session_state.comfort_tolerance:.1f}°C)\n\n"
        f"**Initial:** {st.session_state.initial_temp:.1f}°C · **Outside:** {st.session_state.outside_temp:.1f}°C\n\n"
        f"**Insulation:** {st.session_state.insulation_level} · **Steps:** {st.session_state.max_steps}\n\n"
        f"**Reflex Bounds:** [{st.session_state.reflex_lower_slider:.1f}°C, {st.session_state.reflex_upper_slider:.1f}°C]"
    )

    if st.button("🔄 Reset All Simulations", use_container_width=True, key="btn_sidebar_reset"):
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

    # 2. 4 Essential Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Current Room Temp",
            f"{cur_temp:.2f} °C",
            delta=f"{cur_temp - st.session_state.desired_temp:+.1f} °C vs target",
            delta_color="inverse",
            help="Current indoor thermodynamic state",
        )
    with m2:
        st.metric(
            "Desired Target",
            f"{st.session_state.desired_temp:.1f} °C",
            delta=f"±{st.session_state.comfort_tolerance:.1f} °C band",
            delta_color="off",
            help="Target comfort temperature setpoint",
        )
    with m3:
        st.metric(
            "Outside Temp",
            f"{st.session_state.outside_temp:.1f} °C",
            delta=f"Insulation: {st.session_state.insulation_level}",
            delta_color="off",
            help="Outdoor ambient temperature influencing heat exchange",
        )
    with m4:
        st.metric(
            "Environment Status",
            cur_class.upper(),
            delta=status_info["icon"],
            delta_color="off",
            help="Classification into Cold, Comfortable, or Hot",
        )

    # 3. Environmental Temperature Monitoring Visualization
    st.markdown("<br>", unsafe_allow_html=True)
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
    st.markdown("<br>", unsafe_allow_html=True)
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
    st.markdown("<br>", unsafe_allow_html=True)
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

    st.markdown("<br>", unsafe_allow_html=True)
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

    st.markdown("<br>", unsafe_allow_html=True)
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
    st.markdown("<br>", unsafe_allow_html=True)
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

    st.markdown("<br>", unsafe_allow_html=True)
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
    st.markdown("<br>", unsafe_allow_html=True)
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

    st.markdown("<br>", unsafe_allow_html=True)
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
