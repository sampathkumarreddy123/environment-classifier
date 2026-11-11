"""
Main Streamlit Application for the Environment Classifier and Demonstrator.
Final Production Version: Comprehensive Multi-Agent Academic Simulation Platform.

Sections:
1. Home / Overview
2. Environment Configuration (with Demonstration Scenarios & Room Sandbox)
3. Utility-Based Agent
4. Simple Reflex Agent
5. Agent Comparison
6. Results and Downloads
7. About the Project
"""

import io
import time
from typing import Any, Dict, List, Optional
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
    AGENT_REFLEX,
    AGENT_UTILITY,
    TIE_LABEL,
    ComparisonConfig,
    ComparisonRunner,
    determine_metric_winner,
    determine_settling_winner,
)
from utils import (
    create_agent_temperature_history_chart,
    create_comparison_action_distribution_chart,
    create_comparison_bar_chart,
    create_comparison_classification_chart,
    create_comparison_energy_chart,
    create_comparison_error_chart,
    create_comparison_temperature_chart,
    create_reflex_action_summary_chart,
    create_reflex_temperature_history_chart,
    create_temperature_gauge,
    create_utility_comparison_chart,
    create_utility_factors_breakdown_chart,
    get_comparison_summary_html,
    get_decision_card_html,
    get_reflex_decision_card_html,
    get_reflex_rules_html,
    get_status_card_html,
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Environment Classifier & Demonstrator",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean, professional academic CSS (no excessive gradients, neon glows, or clutter)
st.markdown(
    """
    <style>
    .academic-header {
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.8rem;
        margin-bottom: 1.2rem;
    }
    .academic-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .academic-subtitle {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0369a1;
        margin-bottom: 0.4rem;
    }
    .academic-badge {
        display: inline-block;
        padding: 4px 10px;
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.5rem;
    }
    .desc-box {
        background-color: #f8fafc;
        border-left: 4px solid #0284c7;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 0.95rem;
        color: #334155;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }
    .card-academic {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.15s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
def init_session_state() -> None:
    """Initialize application session state variables with default configurations."""
    # Active scenario tracker
    if "active_scenario_id" not in st.session_state:
        st.session_state.active_scenario_id = "cold_room"

    # Physical environment parameters
    if "initial_temp" not in st.session_state:
        st.session_state.initial_temp = SimulationDefaults.DEFAULT_INITIAL_TEMP
    if "desired_temp" not in st.session_state:
        st.session_state.desired_temp = SimulationDefaults.DEFAULT_DESIRED_TEMP
    if "outside_temp" not in st.session_state:
        st.session_state.outside_temp = SimulationDefaults.DEFAULT_OUTSIDE_TEMP
    if "insulation_level" not in st.session_state:
        st.session_state.insulation_level = "Medium"
    if "comfort_tolerance" not in st.session_state:
        st.session_state.comfort_tolerance = SimulationDefaults.DEFAULT_COMFORT_TOLERANCE
    if "heater_power" not in st.session_state:
        st.session_state.heater_power = SimulationDefaults.DEFAULT_HEATER_POWER
    if "cooling_power" not in st.session_state:
        st.session_state.cooling_power = SimulationDefaults.DEFAULT_COOLING_POWER
    if "disturbance_level" not in st.session_state:
        st.session_state.disturbance_level = SimulationDefaults.DEFAULT_DISTURBANCE
    if "max_steps" not in st.session_state:
        st.session_state.max_steps = SimulationDefaults.DEFAULT_STEPS
    if "random_seed" not in st.session_state:
        st.session_state.random_seed = 42

    # Utility function weights
    if "weight_comfort" not in st.session_state:
        st.session_state.weight_comfort = UtilityWeightsDefaults.DEFAULT_WEIGHT_COMFORT
    if "weight_energy" not in st.session_state:
        st.session_state.weight_energy = UtilityWeightsDefaults.DEFAULT_WEIGHT_ENERGY
    if "weight_stability" not in st.session_state:
        st.session_state.weight_stability = UtilityWeightsDefaults.DEFAULT_WEIGHT_STABILITY
    if "weight_overshoot" not in st.session_state:
        st.session_state.weight_overshoot = UtilityWeightsDefaults.DEFAULT_WEIGHT_OVERSHOOT

    # Simple Reflex Thermostat thresholds
    if "reflex_lower_slider" not in st.session_state:
        st.session_state.reflex_lower_slider = ReflexThermostatDefaults.DEFAULT_LOWER_THRESHOLD
    if "reflex_upper_slider" not in st.session_state:
        st.session_state.reflex_upper_slider = ReflexThermostatDefaults.DEFAULT_UPPER_THRESHOLD

    # Navigation state
    if "nav_selection" not in st.session_state:
        st.session_state.nav_selection = "1. Home / Overview"

    # Simulation instances
    if "env" not in st.session_state:
        reset_utility_simulation()

    if "reflex_env" not in st.session_state:
        reset_reflex_simulation()

    if "comparison_running" not in st.session_state:
        st.session_state.comparison_running = False
    if "comparison_runner" not in st.session_state:
        reset_comparison_simulation()

    if "sandbox_env" not in st.session_state:
        st.session_state.sandbox_env = _create_configured_environment()


def _build_comparison_config() -> ComparisonConfig:
    """Build a complete configuration snapshot for synchronized multi-agent comparison."""
    return ComparisonConfig(
        initial_temp=float(st.session_state.initial_temp),
        desired_temp=float(st.session_state.desired_temp),
        outside_temp=float(st.session_state.outside_temp),
        insulation_level=str(st.session_state.insulation_level),
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


def _create_configured_environment() -> RoomEnvironment:
    """Instantiate a RoomEnvironment using current session state settings."""
    return RoomEnvironment(
        initial_temp=st.session_state.initial_temp,
        desired_temp=st.session_state.desired_temp,
        outside_temp=st.session_state.outside_temp,
        insulation_level=st.session_state.insulation_level,
        heater_power=st.session_state.heater_power,
        cooling_power=st.session_state.cooling_power,
        disturbance_level=st.session_state.disturbance_level,
        comfort_tolerance=st.session_state.comfort_tolerance,
        random_seed=st.session_state.random_seed,
    )


def reset_utility_simulation() -> None:
    """Reset the independent Utility-Based Agent and environment."""
    st.session_state.env = _create_configured_environment()
    st.session_state.agent = UtilityBasedAgent(
        weight_comfort=st.session_state.weight_comfort,
        weight_energy=st.session_state.weight_energy,
        weight_stability=st.session_state.weight_stability,
        weight_overshoot=st.session_state.weight_overshoot,
    )
    init_obs = st.session_state.env.get_state()
    initial_dec = st.session_state.agent.decide_action(init_obs)
    # Clear step-0 tracking
    st.session_state.agent.total_energy_consumed = 0.0
    st.session_state.agent.action_history = []
    st.session_state.agent.decision_history = []

    st.session_state.simulation_records = [{
        "step": 0,
        "room_temp": init_obs["room_temp"],
        "outside_temp": init_obs["outside_temp"],
        "desired_temp": init_obs["desired_temp"],
        "action": ACTION_OFF,
        "predicted_temp": init_obs["room_temp"],
        "actual_temp": init_obs["room_temp"],
        "utility_score": initial_dec.utility_score,
        "energy_consumed": 0.0,
        "classification": init_obs["classification"],
    }]


def reset_reflex_simulation() -> None:
    """Reset the independent Simple Reflex Agent and environment."""
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


def reset_comparison_simulation() -> None:
    """Reset the synchronized multi-agent comparison engine."""
    st.session_state.comparison_running = False
    st.session_state.comparison_runner = ComparisonRunner(_build_comparison_config())


def reset_all_simulations() -> None:
    """Reset all simulation environments, agents, and comparison runners."""
    reset_utility_simulation()
    reset_reflex_simulation()
    reset_comparison_simulation()
    st.session_state.sandbox_env = _create_configured_environment()


def apply_demonstration_scenario(scenario_id: str) -> None:
    """Apply a preconfigured academic demonstration scenario."""
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

# Keep comparison runner synced with latest parameters
latest_cfg = _build_comparison_config()
if comparison_runner.config != latest_cfg:
    comparison_runner.reset(latest_cfg)
    st.session_state.comparison_running = False


# ---------------------------------------------------------
# Sidebar Navigation & Preset Controls
# ---------------------------------------------------------
SECTION_HOME = "1. Home / Overview"
SECTION_CONFIG = "2. Environment Configuration"
SECTION_UTILITY = "3. Utility-Based Agent"
SECTION_REFLEX = "4. Simple Reflex Agent"
SECTION_COMPARISON = "5. Agent Comparison"
SECTION_RESULTS = "6. Results and Downloads"
SECTION_ABOUT = "7. About the Project"
SECTIONS = [
    SECTION_HOME,
    SECTION_CONFIG,
    SECTION_UTILITY,
    SECTION_REFLEX,
    SECTION_COMPARISON,
    SECTION_RESULTS,
    SECTION_ABOUT,
]

with st.sidebar:
    st.markdown("### 🎓 Academic Project Navigator")
    nav_selection = st.radio(
        "Application Section",
        options=SECTIONS,
        key="nav_selection",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("#### 🎯 Demonstration Scenarios")
    scenario_choice = st.selectbox(
        "Load Preset Scenario",
        options=["Scenario 1: Cold Room", "Scenario 2: Hot Room", "Scenario 3: Comfortable Room", "Custom Configuration"],
        index=0 if st.session_state.active_scenario_id == "cold_room" else (
            1 if st.session_state.active_scenario_id == "hot_room" else (
                2 if st.session_state.active_scenario_id == "comfortable_room" else 3
            )
        ),
        help="Quickly load preconfigured academic benchmark scenarios.",
    )

    if st.button("⚡ Apply Selected Scenario", use_container_width=True, type="secondary"):
        if "Scenario 1" in scenario_choice:
            apply_demonstration_scenario("cold_room")
        elif "Scenario 2" in scenario_choice:
            apply_demonstration_scenario("hot_room")
        elif "Scenario 3" in scenario_choice:
            apply_demonstration_scenario("comfortable_room")
        st.success("Scenario applied to all agents!")
        st.rerun()

    st.markdown("---")
    st.markdown("#### 📌 Current Simulation State")
    st.caption(
        f"**Target:** {st.session_state.desired_temp:.1f}°C (±{st.session_state.comfort_tolerance:.1f}°C)\n\n"
        f"**Initial:** {st.session_state.initial_temp:.1f}°C · **Outside:** {st.session_state.outside_temp:.1f}°C\n\n"
        f"**Insulation:** {st.session_state.insulation_level} · **Noise σ:** {st.session_state.disturbance_level:.2f}°C\n\n"
        f"**Reflex Deadband:** [{st.session_state.reflex_lower_slider:.1f}°C, {st.session_state.reflex_upper_slider:.1f}°C]"
    )

    if st.button("🔄 Reset All Simulations", use_container_width=True):
        reset_all_simulations()
        st.rerun()


# =========================================================
# SECTION 1: Home / Overview
# =========================================================
if nav_selection == SECTION_HOME:
    st.markdown(
        """<div class="academic-header">
<div class="academic-badge">Academic Demonstration Platform · Intelligent Control Systems</div>
<div class="academic-title">Environment Classifier and Demonstrator</div>
<div class="academic-subtitle">Simulating and Comparing Utility-Based Optimization vs. Simple Reflex Decision-Making</div>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div class="desc-box">
<b>Executive Summary:</b> This project implements an interactive thermodynamic room-temperature control simulation
to evaluate and compare two fundamental artificial intelligence agent paradigms:
a forward-looking <b>Utility-Based Agent</b> that maximizes a multi-objective utility function, and a <b>Simple Reflex Thermostat</b>
operating strictly on fixed threshold condition-action rules. Both agents are benchmarked under mathematically identical
physical conditions and shared stochastic disturbances.
</div>""",
        unsafe_allow_html=True,
    )

    st.subheader("🏛️ System Architecture")
    col_arch1, col_arch2, col_arch3 = st.columns(3)
    with col_arch1:
        st.markdown(
            """<div class="card-academic">
<h4 style="margin-top:0; color:#0369a1;">1. Thermodynamic Room Model</h4>
Continuous physical environment governed by Newton's Law of Cooling, active HVAC actuation (heating/cooling), wall insulation coefficients, and stochastic Gaussian noise.
</div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<div class="card-academic">
<h4 style="margin-top:0; color:#0369a1;">2. Environment Classifier</h4>
Continuous 3-state temperature classification into <b>Cold</b> (below target band), <b>Comfortable</b> (optimal zone), and <b>Hot</b> (above target band).
</div>""",
            unsafe_allow_html=True,
        )

    with col_arch2:
        st.markdown(
            """<div class="card-academic">
<h4 style="margin-top:0; color:#0284c7;">3. Utility-Based Agent</h4>
Predicts future thermal outcomes using its internal model and selects the action that maximizes a multi-objective utility function balancing comfort, energy cost, stability, and overshoot penalty.
</div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<div class="card-academic">
<h4 style="margin-top:0; color:#b91c1c;">4. Simple Reflex Thermostat</h4>
Pure condition-action reflex agent that actuates HEAT, COOL, or OFF strictly based on current observed room temperature against configured lower and upper threshold bounds.
</div>""",
            unsafe_allow_html=True,
        )

    with col_arch3:
        st.markdown(
            """<div class="card-academic">
<h4 style="margin-top:0; color:#16a34a;">5. Synchronized Benchmark Engine</h4>
Executes both agents side-by-side with identical initial temperatures, weather conditions, and <b>step-for-step identical random noise</b>, computing 11 quantitative performance metrics.
</div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<div class="card-academic">
<h4 style="margin-top:0; color:#475569;">6. Academic Telemetry & Export</h4>
Comprehensive step-by-step telemetry, graphical diagnostics, and complete CSV data export for academic auditing and grading.
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("🧪 Preconfigured Academic Demonstration Scenarios")
    st.caption("Select a scenario to instantly configure all simulation parameters:")

    sc_c1, sc_c2, sc_c3 = st.columns(3)
    with sc_c1:
        st.markdown(
            """<div class="card-academic" style="border-top: 4px solid #0284c7;">
<h4 style="margin-top:0; color:#0284c7;">Scenario 1: Cold Room</h4>
<p style="font-size:0.9rem; color:#475569;"><b>Conditions:</b> 14°C initial temp, 22°C desired target, 10°C outside weather.</p>
<p style="font-size:0.85rem; color:#64748b;"><b>Key Lesson:</b> Demonstrates heating actuation, thermal inertia recovery, and how the utility agent avoids comfort overshooting.</p>
</div>""",
            unsafe_allow_html=True,
        )
        if st.button("Load Scenario 1 & Go to Comparison", key="btn_load_sc1", use_container_width=True):
            apply_demonstration_scenario("cold_room")
            st.session_state.nav_selection = SECTION_COMPARISON
            st.rerun()

    with sc_c2:
        st.markdown(
            """<div class="card-academic" style="border-top: 4px solid #dc2626;">
<h4 style="margin-top:0; color:#dc2626;">Scenario 2: Hot Room</h4>
<p style="font-size:0.9rem; color:#475569;"><b>Conditions:</b> 32°C initial temp, 22°C desired target, 36°C heatwave outside.</p>
<p style="font-size:0.85rem; color:#64748b;"><b>Key Lesson:</b> Demonstrates active cooling response under severe outdoor heat, power consumption, and thermal settling.</p>
</div>""",
            unsafe_allow_html=True,
        )
        if st.button("Load Scenario 2 & Go to Comparison", key="btn_load_sc2", use_container_width=True):
            apply_demonstration_scenario("hot_room")
            st.session_state.nav_selection = SECTION_COMPARISON
            st.rerun()

    with sc_c3:
        st.markdown(
            """<div class="card-academic" style="border-top: 4px solid #16a34a;">
<h4 style="margin-top:0; color:#16a34a;">Scenario 3: Comfortable Room</h4>
<p style="font-size:0.9rem; color:#475569;"><b>Conditions:</b> 22°C initial temp (comfortable), 28°C outside, Good insulation.</p>
<p style="font-size:0.85rem; color:#64748b;"><b>Key Lesson:</b> Highlights energy conservation. The utility agent exploits natural thermal drift to stay OFF without wasting power.</p>
</div>""",
            unsafe_allow_html=True,
        )
        if st.button("Load Scenario 3 & Go to Comparison", key="btn_load_sc3", use_container_width=True):
            apply_demonstration_scenario("comfortable_room")
            st.session_state.nav_selection = SECTION_COMPARISON
            st.rerun()

    st.markdown("---")
    st.subheader("🚀 Quick Navigation")
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    with quick_col1:
        if st.button("⚖️ Jump to Agent Comparison", use_container_width=True, type="primary"):
            st.session_state.nav_selection = SECTION_COMPARISON
            st.rerun()
    with quick_col2:
        if st.button("🤖 Inspect Utility-Based Agent", use_container_width=True):
            st.session_state.nav_selection = SECTION_UTILITY
            st.rerun()
    with quick_col3:
        if st.button("🔁 Inspect Simple Reflex Thermostat", use_container_width=True):
            st.session_state.nav_selection = SECTION_REFLEX
            st.rerun()
    with quick_col4:
        if st.button("📖 Read Theoretical Foundations", use_container_width=True):
            st.session_state.nav_selection = SECTION_ABOUT
            st.rerun()


# =========================================================
# SECTION 2: Environment Configuration & Sandbox
# =========================================================
elif nav_selection == SECTION_CONFIG:
    st.markdown(
        """<div class="academic-header">
<div class="academic-badge">Simulation Parameters · Physical Properties & Weights</div>
<div class="academic-title">Environment & Agent Configuration</div>
<div class="academic-subtitle">Tune Thermodynamic Parameters, Utility Weights, and Thresholds</div>
</div>""",
        unsafe_allow_html=True,
    )

    st.subheader("⚙️ Parameter Controls")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("#### 🌡️ Temperature Targets & Weather")
        st.session_state.initial_temp = st.slider(
            "Initial Room Temperature (°C)",
            min_value=float(SimulationDefaults.MIN_INITIAL_TEMP),
            max_value=float(SimulationDefaults.MAX_INITIAL_TEMP),
            value=float(st.session_state.initial_temp),
            step=0.5,
            help="Starting temperature of the room before simulation begins.",
        )
        st.session_state.desired_temp = st.slider(
            "Desired Comfort Target (°C)",
            min_value=float(SimulationDefaults.MIN_DESIRED_TEMP),
            max_value=float(SimulationDefaults.MAX_DESIRED_TEMP),
            value=float(st.session_state.desired_temp),
            step=0.5,
            help="Target comfort temperature.",
        )
        st.session_state.comfort_tolerance = st.slider(
            "Comfort Tolerance (± °C)",
            min_value=0.5,
            max_value=3.0,
            value=float(st.session_state.comfort_tolerance),
            step=0.1,
            help="Half-width of the acceptable comfort band around the desired target.",
        )
        st.session_state.outside_temp = st.slider(
            "Outside Ambient Temperature (°C)",
            min_value=float(SimulationDefaults.MIN_OUTSIDE_TEMP),
            max_value=float(SimulationDefaults.MAX_OUTSIDE_TEMP),
            value=float(st.session_state.outside_temp),
            step=0.5,
            help="Outdoor temperature driving passive thermal exchange.",
        )

        st.markdown("#### 🏠 Physical Building & HVAC Properties")
        ins_opts = ["Poor", "Medium", "Good"]
        st.session_state.insulation_level = st.selectbox(
            "Room Insulation Quality",
            options=ins_opts,
            index=ins_opts.index(st.session_state.insulation_level),
            help="Poor: k=0.14, Medium: k=0.07, Good: k=0.025.",
        )
        st.session_state.heater_power = st.slider(
            "Heater Actuation Power (°C / step)",
            min_value=float(SimulationDefaults.MIN_HEATER_POWER),
            max_value=float(SimulationDefaults.MAX_HEATER_POWER),
            value=float(st.session_state.heater_power),
            step=0.1,
        )
        st.session_state.cooling_power = st.slider(
            "Cooling Actuation Power (°C / step)",
            min_value=float(SimulationDefaults.MIN_COOLING_POWER),
            max_value=float(SimulationDefaults.MAX_COOLING_POWER),
            value=float(st.session_state.cooling_power),
            step=0.1,
        )
        st.session_state.disturbance_level = st.slider(
            "Environmental Disturbance Noise Level (σ)",
            min_value=float(SimulationDefaults.MIN_DISTURBANCE),
            max_value=float(SimulationDefaults.MAX_DISTURBANCE),
            value=float(st.session_state.disturbance_level),
            step=0.05,
            help="Standard deviation of stochastic Gaussian thermal disturbance.",
        )
        st.session_state.max_steps = st.slider(
            "Simulation Horizon (Time Steps)",
            min_value=int(SimulationDefaults.MIN_STEPS),
            max_value=int(SimulationDefaults.MAX_STEPS),
            value=int(st.session_state.max_steps),
            step=5,
        )
        st.session_state.random_seed = st.number_input(
            "Stochastic Disturbance Seed",
            value=int(st.session_state.random_seed),
            step=1,
            help="Guarantees strictly identical disturbance sequences across runs.",
        )

    with col_p2:
        st.markdown("#### ⚖️ Utility-Based Agent Weights")
        st.markdown(
            "$$\\text{Utility}(a) = - w_c P_{\\text{comfort}} - w_e C_{\\text{energy}} - w_s P_{\\text{instability}} - w_o P_{\\text{overshoot}}$$"
        )
        st.session_state.weight_comfort = st.slider(
            "Comfort Weight (w_c)",
            min_value=0.0,
            max_value=4.0,
            value=float(st.session_state.weight_comfort),
            step=0.1,
            help="Multiplier for proximity to target temperature.",
        )
        st.session_state.weight_energy = st.slider(
            "Energy Cost Weight (w_e)",
            min_value=0.0,
            max_value=3.0,
            value=float(st.session_state.weight_energy),
            step=0.05,
            help="Multiplier penalizing active heating/cooling energy.",
        )
        st.session_state.weight_stability = st.slider(
            "Instability Penalty Weight (w_s)",
            min_value=0.0,
            max_value=2.0,
            value=float(st.session_state.weight_stability),
            step=0.05,
            help="Penalizes abrupt temperature swings between steps.",
        )
        st.session_state.weight_overshoot = st.slider(
            "Overshoot Penalty Weight (w_o)",
            min_value=0.0,
            max_value=3.0,
            value=float(st.session_state.weight_overshoot),
            step=0.1,
            help="Heavily penalizes projecting outside the acceptable comfort boundaries.",
        )
        agent.update_weights(
            weight_comfort=st.session_state.weight_comfort,
            weight_energy=st.session_state.weight_energy,
            weight_stability=st.session_state.weight_stability,
            weight_overshoot=st.session_state.weight_overshoot,
        )

        st.markdown("#### 🔁 Simple Reflex Thermostat Thresholds")

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
            "Lower Temperature Threshold (°C)",
            min_value=float(ReflexThermostatDefaults.MIN_THRESHOLD),
            max_value=float(ReflexThermostatDefaults.MAX_THRESHOLD - 0.5),
            step=0.5,
            key="reflex_lower_slider",
            on_change=_sync_reflex_lower,
            help="Temperature below which HEAT is triggered.",
        )
        st.slider(
            "Upper Temperature Threshold (°C)",
            min_value=float(ReflexThermostatDefaults.MIN_THRESHOLD + 0.5),
            max_value=float(ReflexThermostatDefaults.MAX_THRESHOLD),
            step=0.5,
            key="reflex_upper_slider",
            on_change=_sync_reflex_upper,
            help="Temperature above which COOL is triggered.",
        )
        st.caption(
            f"**Active Inactive Deadband (OFF):** [{st.session_state.reflex_lower_slider:.1f}°C, "
            f"{st.session_state.reflex_upper_slider:.1f}°C]"
        )
        reflex_agent.update_thresholds(
            lower_threshold=st.session_state.reflex_lower_slider,
            upper_threshold=st.session_state.reflex_upper_slider,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Apply Configuration to All Agents & Reset", use_container_width=True, type="primary"):
            reset_all_simulations()
            st.success("All simulations re-initialized with new configuration!")
            st.rerun()

    st.markdown("---")
    # Interactive Sandbox sub-section
    st.subheader("🧪 Interactive Room Sandbox (Manual Thermodynamic Testing)")
    st.caption("Inspect passive thermal drift and test manual HVAC commands without any agent intervention.")

    s_env: RoomEnvironment = st.session_state.sandbox_env
    s_state = s_env.get_state()
    s_history = s_env.get_history_df()

    st.markdown(
        get_status_card_html(
            classification=s_state["classification"],
            current_temp=s_state["room_temp"],
            desired_temp=s_state["desired_temp"],
        ),
        unsafe_allow_html=True,
    )

    sb_col1, sb_col2, sb_col3 = st.columns([2, 1, 1])
    with sb_col1:
        sandbox_act = st.radio(
            "Manual Actuation:",
            options=[ACTION_OFF, ACTION_HEAT, ACTION_COOL],
            horizontal=True,
            format_func=lambda x: {
                ACTION_OFF: "⏸️ OFF (Natural Drift)",
                ACTION_HEAT: "🔥 HEAT (+Heater)",
                ACTION_COOL: "❄️ COOL (-Cooler)",
            }[x],
            key="sandbox_manual_action",
        )
    with sb_col2:
        if st.button("▶️ Step Sandbox", key="btn_step_sandbox_cfg", use_container_width=True):
            s_env.step(sandbox_act)
            st.rerun()
    with sb_col3:
        if st.button("↩️ Reset Sandbox", key="btn_reset_sandbox_cfg", use_container_width=True):
            s_env.reset()
            st.rerun()

    s_g1, s_g2 = st.columns(2)
    with s_g1:
        st.plotly_chart(
            create_temperature_gauge(
                current_temp=s_state["room_temp"],
                desired_temp=s_state["desired_temp"],
                comfort_tolerance=s_env.comfort_tolerance,
            ),
            use_container_width=True,
            key="chart_sandbox_gauge_config",
        )
    with s_g2:
        st.plotly_chart(
            create_comparison_bar_chart(
                room_temp=s_state["room_temp"],
                desired_temp=s_state["desired_temp"],
                outside_temp=s_state["outside_temp"],
            ),
            use_container_width=True,
            key="chart_sandbox_bar_config",
        )


# =========================================================
# SECTION 3: Utility-Based Agent
# =========================================================
elif nav_selection == SECTION_UTILITY:
    st.markdown(
        """<div class="academic-header">
<div class="academic-badge">Agent Paradigm 1 · Multi-Objective Optimization</div>
<div class="academic-title">Utility-Based Intelligent Agent</div>
<div class="academic-subtitle">Dynamic Thermal State Projection & Multi-Factor Utility Maximization</div>
</div>""",
        unsafe_allow_html=True,
    )

    current_state = env.get_state()
    last_dec = agent.last_decision

    # Status card
    st.markdown(
        get_status_card_html(
            classification=current_state["classification"],
            current_temp=current_state["room_temp"],
            desired_temp=current_state["desired_temp"],
        ),
        unsafe_allow_html=True,
    )

    # Key Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        delta_target = current_state["room_temp"] - current_state["desired_temp"]
        st.metric(
            label="Current Room Temp",
            value=f"{current_state['room_temp']:.2f} °C",
            delta=f"{delta_target:+.2f} °C vs Target",
            delta_color="inverse",
        )
    with m2:
        st.metric(
            label="Target Comfort Target",
            value=f"{current_state['desired_temp']:.1f} °C",
            help=f"Acceptable band: {current_state['desired_temp'] - env.comfort_tolerance:.1f}°C to {current_state['desired_temp'] + env.comfort_tolerance:.1f}°C",
        )
    with m3:
        st.metric(
            label="Outside Temperature",
            value=f"{current_state['outside_temp']:.1f} °C",
            delta=f"{current_state['outside_temp'] - current_state['room_temp']:+.1f} °C ambient delta",
            delta_color="off",
        )
    with m4:
        st.metric(
            label="Environment Status",
            value=f"{current_state['classification']}",
        )
    with m5:
        progress_pct = min(100.0, (env.current_step / st.session_state.max_steps) * 100.0)
        st.metric(
            label="Simulation Progress",
            value=f"Step {env.current_step} / {st.session_state.max_steps}",
            delta=f"{progress_pct:.0f}% completed",
        )

    st.progress(progress_pct / 100.0)
    st.markdown("<br>", unsafe_allow_html=True)

    # Controls
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
        if st.button("▶️ Step 1 Step", key="btn_u_step", use_container_width=True):
            if env.current_step < st.session_state.max_steps:
                perform_agent_step()
                st.rerun()
            else:
                st.warning("Horizon reached. Reset to restart.")
    with c2:
        if st.button("⚡ Run Full Simulation", key="btn_u_run", use_container_width=True, type="primary"):
            remaining = st.session_state.max_steps - env.current_step
            if remaining > 0:
                for _ in range(remaining):
                    perform_agent_step()
                st.rerun()
            else:
                st.warning("Horizon reached. Reset to restart.")
    with c3:
        if st.button("↩️ Reset Simulation", key="btn_u_reset", use_container_width=True):
            reset_utility_simulation()
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Decision intelligence
    st.subheader("🧠 Agent Decision Intelligence & Reasoning")
    if last_dec:
        st.markdown(
            get_decision_card_html(
                current_temp=last_dec.current_temp,
                desired_temp=last_dec.desired_temp,
                selected_action=last_dec.selected_action,
                predicted_temp=last_dec.predicted_temp,
                utility_score=last_dec.utility_score,
                reason=last_dec.reason,
            ),
            unsafe_allow_html=True,
        )

        ak1, ak2, ak3, ak4 = st.columns(4)
        with ak1:
            st.metric("Selected Action", f"{last_dec.selected_action}")
        with ak2:
            st.metric("Predicted Next Temp", f"{last_dec.predicted_temp:.2f} °C")
        with ak3:
            st.metric("Utility Score", f"{last_dec.utility_score:.2f}")
        with ak4:
            st.metric("Cumulative Energy", f"{agent.total_energy_consumed:.1f} units")

        u_col1, u_col2 = st.columns(2)
        with u_col1:
            st.plotly_chart(
                create_utility_comparison_chart(
                    action_evaluations=last_dec.action_evaluations,
                    selected_action=last_dec.selected_action,
                ),
                use_container_width=True,
                key="chart_sec_u_comp",
            )
        with u_col2:
            st.plotly_chart(
                create_utility_factors_breakdown_chart(
                    action_evaluations=last_dec.action_evaluations,
                ),
                use_container_width=True,
                key="chart_sec_u_breakdown",
            )

    st.subheader("📈 Temperature Trajectory History")
    sim_df = pd.DataFrame(st.session_state.simulation_records)
    st.plotly_chart(
        create_agent_temperature_history_chart(
            df=sim_df,
            desired_temp=st.session_state.desired_temp,
            comfort_tolerance=st.session_state.comfort_tolerance,
        ),
        use_container_width=True,
        key="chart_sec_u_history",
    )

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(
            create_temperature_gauge(
                current_temp=current_state["room_temp"],
                desired_temp=current_state["desired_temp"],
                comfort_tolerance=env.comfort_tolerance,
            ),
            use_container_width=True,
            key="chart_sec_u_gauge",
        )
    with g2:
        st.plotly_chart(
            create_comparison_bar_chart(
                room_temp=current_state["room_temp"],
                desired_temp=current_state["desired_temp"],
                outside_temp=current_state["outside_temp"],
            ),
            use_container_width=True,
            key="chart_sec_u_bar",
        )

    st.subheader("📋 Telemetry Log")
    if not sim_df.empty:
        st.dataframe(sim_df.tail(12), use_container_width=True, hide_index=True)
        csv_bytes = sim_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Utility Agent Telemetry (CSV)",
            data=csv_bytes,
            file_name="utility_agent_telemetry.csv",
            mime="text/csv",
            key="btn_sec_u_csv",
        )


# =========================================================
# SECTION 4: Simple Reflex Agent
# =========================================================
elif nav_selection == SECTION_REFLEX:
    st.markdown(
        """<div class="academic-header">
<div class="academic-badge">Agent Paradigm 2 · Pure Condition-Action Rules</div>
<div class="academic-title">Simple Reflex Thermostat Agent</div>
<div class="academic-subtitle">Deterministic Rule Execution Over Current Room Temperature Percept</div>
</div>""",
        unsafe_allow_html=True,
    )

    r_state = reflex_env.get_state()
    r_preview = reflex_agent.preview_decision(r_state)
    last_r_dec = reflex_agent.last_decision

    st.markdown(
        get_status_card_html(
            classification=r_state["classification"],
            current_temp=r_state["room_temp"],
            desired_temp=r_state["desired_temp"],
        ),
        unsafe_allow_html=True,
    )

    # Key Metrics
    rm1, rm2, rm3, rm4, rm5, rm6 = st.columns(6)
    with rm1:
        st.metric(
            "Current Room Temp",
            f"{r_state['room_temp']:.2f} °C",
            delta=f"{r_state['room_temp'] - r_state['desired_temp']:+.2f} °C vs Target",
            delta_color="inverse",
        )
    with rm2:
        st.metric("Lower Threshold", f"{reflex_agent.lower_threshold:.1f} °C")
    with rm3:
        st.metric("Upper Threshold", f"{reflex_agent.upper_threshold:.1f} °C")
    with rm4:
        st.metric("Selected Action", f"{r_preview.selected_action}")
    with rm5:
        st.metric("Classification", f"{r_state['classification']}")
    with rm6:
        st.metric("Energy Used", f"{reflex_agent.total_energy_consumed:.1f} units")

    r_progress = min(100.0, (reflex_env.current_step / st.session_state.max_steps) * 100.0)
    st.progress(r_progress / 100.0)
    st.markdown("<br>", unsafe_allow_html=True)

    # Controls
    def perform_reflex_step() -> None:
        if reflex_env.current_step < st.session_state.max_steps:
            rec = reflex_agent.step_environment(reflex_env)
            st.session_state.reflex_simulation_records.append(rec)

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        if st.button("▶️ Step 1 Step", key="btn_sec_r_step", use_container_width=True):
            if reflex_env.current_step < st.session_state.max_steps:
                perform_reflex_step()
                st.rerun()
            else:
                st.warning("Horizon reached. Reset to restart.")
    with rc2:
        if st.button("⚡ Run Full Simulation", key="btn_sec_r_run", use_container_width=True, type="primary"):
            remaining = st.session_state.max_steps - reflex_env.current_step
            if remaining > 0:
                for _ in range(remaining):
                    perform_reflex_step()
                st.rerun()
            else:
                st.warning("Horizon reached. Reset to restart.")
    with rc3:
        if st.button("↩️ Reset Simulation", key="btn_sec_r_reset", use_container_width=True):
            reset_reflex_simulation()
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔁 Condition-Action Rules Evaluation")

    st.markdown(
        get_reflex_rules_html(
            current_temp=r_state["room_temp"],
            lower_threshold=reflex_agent.lower_threshold,
            upper_threshold=reflex_agent.upper_threshold,
            selected_action=r_preview.selected_action,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        get_reflex_decision_card_html(
            current_temp=r_state["room_temp"],
            lower_threshold=reflex_agent.lower_threshold,
            upper_threshold=reflex_agent.upper_threshold,
            selected_action=r_preview.selected_action,
            classification=r_state["classification"],
            step_energy=last_r_dec.energy_consumed if last_r_dec else 0.0,
            total_energy=reflex_agent.total_energy_consumed,
            rule_triggered=r_preview.rule_triggered,
            reason=r_preview.reason,
            last_step_action=last_r_dec.selected_action if last_r_dec else None,
            last_step_num=reflex_env.current_step,
        ),
        unsafe_allow_html=True,
    )

    st.subheader("📈 Temperature Trajectory & Threshold Tracking")
    r_df = pd.DataFrame(st.session_state.reflex_simulation_records)
    st.plotly_chart(
        create_reflex_temperature_history_chart(
            df=r_df,
            lower_threshold=reflex_agent.lower_threshold,
            upper_threshold=reflex_agent.upper_threshold,
            desired_temp=st.session_state.desired_temp,
        ),
        use_container_width=True,
        key="chart_sec_r_history",
    )

    rg1, rg2 = st.columns(2)
    with rg1:
        st.plotly_chart(
            create_temperature_gauge(
                current_temp=r_state["room_temp"],
                desired_temp=r_state["desired_temp"],
                comfort_tolerance=reflex_env.comfort_tolerance,
            ),
            use_container_width=True,
            key="chart_sec_r_gauge",
        )
    with rg2:
        st.plotly_chart(
            create_reflex_action_summary_chart(df=r_df),
            use_container_width=True,
            key="chart_sec_r_pie",
        )

    st.subheader("📋 Telemetry Log")
    if not r_df.empty:
        st.dataframe(r_df.tail(12), use_container_width=True, hide_index=True)
        csv_bytes = r_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Reflex Agent Telemetry (CSV)",
            data=csv_bytes,
            file_name="reflex_agent_telemetry.csv",
            mime="text/csv",
            key="btn_sec_r_csv",
        )


# =========================================================
# SECTION 5: Agent Comparison (Synchronized Benchmark)
# =========================================================
elif nav_selection == SECTION_COMPARISON:
    st.markdown(
        """<div class="academic-header">
<div class="academic-badge">Academic Head-to-Head Benchmark · Identical Disturbances</div>
<div class="academic-title">Multi-Agent Performance Comparison</div>
<div class="academic-subtitle">Synchronized Benchmark: Utility-Based Agent vs. Simple Reflex Thermostat</div>
</div>""",
        unsafe_allow_html=True,
    )

    cfg = comparison_runner.config
    st.caption(
        f"**Shared Synchronized Conditions:** Initial: {cfg.initial_temp:.1f}°C · Target: {cfg.desired_temp:.1f}°C "
        f"(±{cfg.comfort_tolerance:.1f}°C) · Outside: {cfg.outside_temp:.1f}°C · Insulation: {cfg.insulation_level} · "
        f"Disturbance σ: {cfg.disturbance_level:.2f}°C · Shared Seed: {cfg.random_seed} · Reflex Deadband: [{cfg.lower_threshold:.1f}°C, {cfg.upper_threshold:.1f}°C]"
    )

    # Comparison Controls
    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    with cc1:
        run_lbl = "▶️ Resume" if (comparison_runner.current_step > 0 and not comparison_runner.is_finished) else "▶️ Run Comparison"
        if st.button(run_lbl, key="btn_sec_c_run", use_container_width=True, type="primary"):
            if comparison_runner.is_finished:
                comparison_runner.reset(_build_comparison_config())
            st.session_state.comparison_running = True
            st.rerun()
    with cc2:
        if st.button("⏸️ Pause", key="btn_sec_c_pause", use_container_width=True, disabled=not st.session_state.comparison_running):
            st.session_state.comparison_running = False
            st.rerun()
    with cc3:
        if st.button("⚡ Run Instant", key="btn_sec_c_instant", use_container_width=True):
            st.session_state.comparison_running = False
            if comparison_runner.is_finished:
                comparison_runner.run_all()
            else:
                comparison_runner.run_remaining()
            st.rerun()
    with cc4:
        if st.button("🔁 Re-run (Same Seed)", key="btn_sec_c_rerun", use_container_width=True):
            st.session_state.comparison_running = False
            comparison_runner.run_all()
            st.rerun()
    with cc5:
        if st.button("↩️ Reset Comparison", key="btn_sec_c_reset", use_container_width=True):
            reset_comparison_simulation()
            st.rerun()

    comp_prog = min(100.0, (comparison_runner.current_step / comparison_runner.max_steps) * 100.0) if comparison_runner.max_steps > 0 else 0.0
    st.progress(comp_prog / 100.0, text=f"Step {comparison_runner.current_step} of {comparison_runner.max_steps} ({comp_prog:.0f}%)")

    eval_res = comparison_runner.evaluate()
    u_df = eval_res["utility_df"]
    r_df = eval_res["reflex_df"]
    combined_df = eval_res["combined_df"]
    u_metrics = eval_res["utility_metrics"]
    r_metrics = eval_res["reflex_metrics"]
    comp_table_df = eval_res["comparison_df"]
    comp_summary = eval_res["summary"]

    st.markdown(get_comparison_summary_html(comp_summary), unsafe_allow_html=True)

    # Side-by-side overview
    ov1, ov2 = st.columns(2)
    with ov1:
        st.info(
            f"**🤖 Utility-Based Agent** · Final Temp: **{u_df['actual_temp'].iloc[-1]:.2f}°C** "
            f"({u_df['classification'].iloc[-1]}) · Comfort Ratio: **{u_metrics.comfort_percentage:.1f}%** · "
            f"Total Energy: **{u_metrics.total_energy:.1f} units**"
        )
    with ov2:
        st.warning(
            f"**🔁 Simple Reflex Agent** · Final Temp: **{r_df['actual_temp'].iloc[-1]:.2f}°C** "
            f"({r_df['classification'].iloc[-1]}) · Comfort Ratio: **{r_metrics.comfort_percentage:.1f}%** · "
            f"Total Energy: **{r_metrics.total_energy:.1f} units**"
        )

    # Trajectory chart
    st.subheader("📈 Synchronized Dual Temperature Trajectory")
    st.plotly_chart(
        create_comparison_temperature_chart(
            df=combined_df,
            desired_temp=cfg.desired_temp,
            comfort_tolerance=cfg.comfort_tolerance,
            lower_threshold=cfg.lower_threshold,
            upper_threshold=cfg.upper_threshold,
        ),
        use_container_width=True,
        key="chart_sec_c_traj",
    )

    # Quantitative 11-Metric Table
    st.subheader("📊 Quantitative Performance Evaluation Table")
    st.dataframe(comp_table_df, use_container_width=True, hide_index=True)

    # Detailed Analytical Charts
    st.subheader("🔬 Analytical Diagnostics")
    c_tab1, c_tab2, c_tab3, c_tab4 = st.tabs([
        "Action Distribution",
        "Step Error Analysis",
        "Cumulative Energy",
        "Classification Proportions",
    ])
    with c_tab1:
        st.plotly_chart(
            create_comparison_action_distribution_chart(u_metrics=u_metrics, r_metrics=r_metrics),
            use_container_width=True,
            key="chart_sec_c_actions",
        )
    with c_tab2:
        st.plotly_chart(
            create_comparison_error_chart(df=combined_df),
            use_container_width=True,
            key="chart_sec_c_errors",
        )
    with c_tab3:
        st.plotly_chart(
            create_comparison_energy_chart(df=combined_df),
            use_container_width=True,
            key="chart_sec_c_energy",
        )
    with c_tab4:
        st.plotly_chart(
            create_comparison_classification_chart(u_metrics=u_metrics, r_metrics=r_metrics),
            use_container_width=True,
            key="chart_sec_c_classes",
        )

    # Auto-play loop if active
    if st.session_state.comparison_running and not comparison_runner.is_finished:
        time.sleep(0.04)
        comparison_runner.step()
        st.rerun()
    elif comparison_runner.is_finished:
        st.session_state.comparison_running = False


# =========================================================
# SECTION 6: Results and Downloads
# =========================================================
elif nav_selection == SECTION_RESULTS:
    st.markdown(
        """<div class="academic-header">
<div class="academic-badge">Academic Telemetry · CSV Export & Data Integrity</div>
<div class="academic-title">Results and Downloads</div>
<div class="academic-subtitle">Export Simulation Data, Review Statistical Metrics, and Audit Invariants</div>
</div>""",
        unsafe_allow_html=True,
    )

    eval_res = comparison_runner.evaluate()
    comb_df = eval_res["combined_df"]
    u_df = eval_res["utility_df"]
    r_df = eval_res["reflex_df"]
    comp_table = eval_res["comparison_df"]
    u_m = eval_res["utility_metrics"]
    r_m = eval_res["reflex_metrics"]

    st.subheader("✅ Data Integrity & Parity Audit")
    ck1, ck2, ck3, ck4 = st.columns(4)
    with ck1:
        steps_match = len(u_df) == len(r_df)
        st.metric("Steps Synchronized", "PASS" if steps_match else "FAIL", f"{len(comb_df) - 1} steps")
    with ck2:
        dist_match = comb_df["utility_disturbance"].equals(comb_df["reflex_disturbance"]) if "utility_disturbance" in comb_df.columns else True
        st.metric("Disturbance Parity", "PASS (100% Identical)" if dist_match else "FAIL")
    with ck3:
        nan_check = not comb_df.isna().any().any()
        st.metric("Numerical Sanity", "PASS (0 NaNs)" if nan_check else "FAIL")
    with ck4:
        winner_label = eval_res["summary"]["overall_winner"]
        st.metric("Benchmark Winner", winner_label)

    st.markdown("---")
    st.subheader("📥 Export Center")
    st.caption("Download complete experimental datasets in standard CSV format for offline statistical analysis or grading.")

    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        st.download_button(
            "📥 Combined Telemetry (CSV)",
            data=comb_df.to_csv(index=False).encode("utf-8"),
            file_name="comparison_telemetry_combined.csv",
            mime="text/csv",
            key="dl_comb_csv",
            use_container_width=True,
        )
    with d_col2:
        st.download_button(
            "📥 Utility Agent Log (CSV)",
            data=u_df.to_csv(index=False).encode("utf-8"),
            file_name="utility_agent_telemetry.csv",
            mime="text/csv",
            key="dl_u_csv",
            use_container_width=True,
        )
    with d_col3:
        st.download_button(
            "📥 Reflex Agent Log (CSV)",
            data=r_df.to_csv(index=False).encode("utf-8"),
            file_name="reflex_agent_telemetry.csv",
            mime="text/csv",
            key="dl_r_csv",
            use_container_width=True,
        )
    with d_col4:
        st.download_button(
            "📥 Performance Summary (CSV)",
            data=comp_table.to_csv(index=False).encode("utf-8"),
            file_name="benchmark_metrics_summary.csv",
            mime="text/csv",
            key="dl_metrics_csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("📋 Consolidated Telemetry Table")
    st.dataframe(comb_df, use_container_width=True, hide_index=True)

    st.subheader("📊 Metric Comparison Snapshot")
    st.dataframe(comp_table, use_container_width=True, hide_index=True)


# =========================================================
# SECTION 7: About the Project
# =========================================================
elif nav_selection == SECTION_ABOUT:
    st.markdown(
        """<div class="academic-header">
<div class="academic-badge">Academic Curriculum · Theoretical Foundations & Viva Q&A</div>
<div class="academic-title">About the Project & Theoretical Foundations</div>
<div class="academic-subtitle">Artificial Intelligence, Agent Paradigms, Thermodynamics, and Control Theory</div>
</div>""",
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4 = st.tabs([
        "1. Intelligent Agents & Classifiers",
        "2. Agent Decision Architectures",
        "3. Mathematical Thermodynamic Model",
        "4. Comparative Analysis & Viva Q&A",
    ])

    with t1:
        st.markdown(
            """
            ### 🤖 What is an Intelligent Agent?
            In artificial intelligence (Russell & Norvig), an **agent** is an autonomous entity that:
            - **Perceives** its environment through sensors (e.g., room temperature sensor, outdoor ambient sensor).
            - **Reasons** over those percepts based on its internal knowledge, rules, or utility formulation.
            - **Acts** upon the environment via actuators (e.g., HVAC heating compressors, cooling chillers, passive drift).

            A **rational agent** is one that selects an action expected to maximize its performance measure, given its percept history and built-in knowledge.

            ---

            ### 🏷️ What is an Environment Classifier?
            An **environment classifier** is a supervisory perception subsystem that continuously categorizes the state of the physical environment into discrete semantic comfort categories:

            $$\\text{State}(T) = \\begin{cases}
            \\textbf{Cold}, & T < T_{\\text{desired}} - \\tau \\\\
            \\textbf{Comfortable}, & T_{\\text{desired}} - \\tau \\le T \\le T_{\\text{desired}} + \\tau \\\\
            \\textbf{Hot}, & T > T_{\\text{desired}} + \\tau
            \\end{cases}$$

            Where $T_{\\text{desired}}$ is the comfort target and $\\tau$ is the acceptable comfort tolerance (e.g., $\\pm 1.5^\\circ\\text{C}$).
            This classification provides real-time feedback on whether the climate control agent is fulfilling human thermal comfort expectations.
            """
        )

    with t2:
        st.markdown(
            """
            ### 🔁 What is a Simple Reflex Agent?
            A **Simple Reflex Agent** is a reactive control system that bases its decisions strictly on the **current percept**, completely ignoring the percept history, future projections, and energy costs. It maps conditions directly to actions via condition-action rules:

            $$\\text{Action}(T_t) = \\begin{cases}
            \\text{HEAT}, & T_t < T_{\\text{lower}} \\\\
            \\text{COOL}, & T_t > T_{\\text{upper}} \\\\
            \\text{OFF}, & T_{\\text{lower}} \\le T_t \\le T_{\\text{upper}}
            \\end{cases}$$

            - **Advantages:** Computationally minimal ($O(1)$ lookup), highly reliable, zero state modeling overhead.
            - **Limitations:** Suffers from threshold hunting (rapid chattering around boundaries), cannot foresee severe ambient drift, and wastes energy by heating/cooling at full power right up to the threshold.

            ---

            ### 🧠 What is a Utility-Based Agent?
            A **Utility-Based Agent** goes beyond binary goals. It maintains an **internal thermodynamic model** of the world to project future states $\\hat{T}_{t+1}(a)$ for each candidate action $a \\in \\{\\text{HEAT}, \\text{COOL}, \\text{OFF}\\}$, and evaluates each action using an explicit mathematical **utility function**:

            $$a^* = \\arg\\max_{a \\in \\{\\text{HEAT}, \\text{COOL}, \\text{OFF}\\}} \\text{Utility}(a)$$

            #### The Multi-Objective Utility Function:
            $$\\text{Utility}(a) = - \\left( w_c \\cdot P_{\\text{comfort}} + w_e \\cdot C_{\\text{energy}} + w_s \\cdot P_{\\text{instability}} + w_o \\cdot P_{\\text{overshoot}} \\right)$$

            1. **Comfort Penalty ($P_{\\text{comfort}}$):** Proximity to target: $P_{\\text{comfort}} = |\\hat{T}_{t+1}(a) - T_{\\text{desired}}|$.
            2. **Energy Cost ($C_{\\text{energy}}$):** Electrical power consumed: $C_{\\text{energy}} = 1.0$ for HEAT/COOL, $0.0$ for OFF.
            3. **Instability Penalty ($P_{\\text{instability}}$):** Abrupt step-to-step temperature changes: $P_{\\text{instability}} = |\\hat{T}_{t+1}(a) - T_t|$.
            4. **Overshoot Penalty ($P_{\\text{overshoot}}$):** Distance exceeding the comfort band $[T_{\\text{desired}} - \\tau, T_{\\text{desired}} + \\tau]$.
            """
        )

    with t3:
        st.markdown(
            """
            ### 🧮 Mathematical Thermodynamic Environment Model
            The room temperature simulation is modeled using a discrete thermodynamic heat-transfer equation derived from **Newton's Law of Cooling**:

            $$T_{t+1} = T_t + \\Delta T_{\\text{ambient}} + \\Delta T_{\\text{HVAC}}(a) + \\epsilon_t$$

            #### 1. Passive Ambient Heat Transfer:
            $$\\Delta T_{\\text{ambient}} = k_{\\text{loss}} \\cdot (T_{\\text{outside}} - T_t)$$
            Where $k_{\\text{loss}}$ represents the overall heat loss coefficient of the building envelope:
            - **Poor Insulation:** $k = 0.14$ (single pane windows, uninsulated walls).
            - **Medium Insulation:** $k = 0.07$ (standard double-glazed construction).
            - **Good Insulation:** $k = 0.025$ (high-performance airtight thermal envelope).

            #### 2. Active HVAC Actuation:
            $$\\Delta T_{\\text{HVAC}}(a) = \\begin{cases}
            +P_{\\text{heater}}, & \\text{if } a = \\text{HEAT} \\\\
            -P_{\\text{cooling}}, & \\text{if } a = \\text{COOL} \\\\
            0, & \\text{if } a = \\text{OFF}
            \\end{cases}$$

            #### 3. Stochastic Environmental Disturbance:
            $$\\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2)$$
            Models random thermal shocks such as doors opening, sudden solar radiation spikes, or changing occupant body heat.
            """
        )

    with t4:
        st.markdown(
            """
            ### ⚖️ Head-to-Head Architectural Comparison

            | Feature / Dimension | Simple Reflex Thermostat | Utility-Based Intelligent Agent |
            | :--- | :--- | :--- |
            | **Decision Basis** | Current temperature percept $T_t$ only | Projected state $\\hat{T}_{t+1}(a)$ & utility scoring |
            | **Internal World Model** | None (Stateless / Model-Free) | Thermodynamic Newton heat loss model |
            | **Lookahead Capability** | None ($0$ steps) | $1$-step forward projection |
            | **Energy Awareness** | Blind (turns on full power if past threshold) | Explicitly penalizes power consumption ($w_e C_e$) |
            | **Thermal Overshoot** | High (frequently overshoots comfort band) | Minimal (heavily penalizes boundary exits) |
            | **Chattering / Cycling** | Prone to hunting around threshold limits | Highly damped via instability penalty ($w_s P_s$) |
            | **Computational Cost** | $O(1)$ constant time | $O(|A|)$ evaluations across all actions |
            | **Complexity & Tuning** | Easy (requires 2 thresholds) | Requires balancing 4 objective weights |

            ---

            ### 🎓 Sample Oral Defense / Viva Q&A
            **Q1: Why does the Reflex Thermostat suffer from overshoot?**
            > **Answer:** The reflex thermostat operates without knowledge of environmental inertia. It continues heating or cooling at full capacity right until the threshold temperature is crossed. Even after turning off, thermal momentum and outdoor ambient heat flux cause the room to overshoot the comfort zone.

            **Q2: How does the Utility Agent prevent unnecessary energy consumption?**
            > **Answer:** When the room is near the comfort target and the outside temperature is favorable, the utility agent evaluates that turning OFF incurs zero energy cost ($C_{\\text{energy}} = 0$) while passive drift will maintain comfort. The net utility of OFF exceeds HEAT or COOL, saving significant power.

            **Q3: Why must both agents receive identical disturbance sequences in comparison?**
            > **Answer:** In scientific benchmarking, the control algorithm is the sole independent variable. Giving both agents identical initial conditions, identical outside temperatures, and bit-for-bit identical stochastic disturbances guarantees that performance differences reflect the agents' decision-making capabilities rather than random weather variations.
            """
        )
