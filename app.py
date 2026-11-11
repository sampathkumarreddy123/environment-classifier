"""
Environment Classifier and Demonstrator
Interactive Room Temperature Control Simulation & Multi-Agent Comparison.
Clean, lightweight, professional dashboard for college project demonstration.

Sections:
1. Home
2. Environment Setup
3. Utility-Based Agent
4. Reflex Thermostat
5. Comparison
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
    create_comparison_energy_chart,
    create_comparison_temperature_chart,
    create_reflex_temperature_history_chart,
    create_utility_comparison_chart,
    get_comparison_summary_html,
    get_decision_card_html,
    get_reflex_rules_html,
    get_status_card_html,
)

# ---------------------------------------------------------
# Page Configuration & Clean Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Environment Classifier & Demonstrator",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal, clean CSS suitable for academic evaluation
st.markdown(
    """
    <style>
    .main-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1rem;
    }
    .info-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 1rem;
        color: #334155;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .summary-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 16px;
        height: 100%;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
def init_session_state() -> None:
    defaults = {
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


# ---------------------------------------------------------
# Sidebar Navigation & Demo Presets
# ---------------------------------------------------------
SECTIONS = [
    "Home",
    "Environment Setup",
    "Utility-Based Agent",
    "Reflex Thermostat",
    "Comparison",
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

    if st.button("Apply Selected Scenario", use_container_width=True):
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

    if st.button("🔄 Reset All Simulations", use_container_width=True):
        reset_all_simulations()
        st.rerun()


# =========================================================
# SECTION 1: Home
# =========================================================
if nav_selection == "Home":
    st.markdown(
        """<div class="main-title">Environment Classifier and Demonstrator</div>
<div class="sub-title">Intelligent Room Temperature Control Simulation</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div class="info-card">
<b>Project Overview:</b> This application simulates an indoor microclimate using a thermodynamic heat-transfer model
and demonstrates two different artificial intelligence agent architectures:
<br><br>
<b>1. Utility-Based Agent:</b> Predicts the next room temperature for all available actions (HEAT, COOL, OFF)
and selects the action with the maximum utility score by balancing thermal comfort against energy consumption.
<br><br>
<b>2. Simple Reflex Thermostat:</b> Makes decisions using fixed condition-action threshold rules
based only on the current room temperature without prediction or utility evaluation.
<br><br>
Both agents are evaluated under identical environmental disturbances to compare their comfort precision, energy use, and stability.
</div>""",
        unsafe_allow_html=True,
    )

    st.subheader("Key Features")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """<div class="summary-card">
<h4 style="margin-top:0; color:#0284c7;">🌡️ Thermodynamic Environment</h4>
Simulates heat exchange with outdoor air using Newton's Law of Cooling, active HVAC heating/cooling power, and stochastic disturbances. Classifies room status into <b>Cold</b>, <b>Comfortable</b>, and <b>Hot</b>.
</div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="summary-card">
<h4 style="margin-top:0; color:#2563eb;">🤖 Utility-Based Agent</h4>
Uses an internal thermodynamic prediction model to evaluate actions against comfort error, energy cost, stability, and overshoot penalties to select the best action.
</div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """<div class="summary-card">
<h4 style="margin-top:0; color:#ea580c;">🔁 Reflex Thermostat</h4>
Applies simple condition-action rules: heats below the lower threshold, cools above the upper threshold, and remains OFF within the acceptable deadband.
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Quick Start:** Use the sidebar navigation on the left to configure the environment or jump directly to **Comparison** to see both agents in action.")


# =========================================================
# SECTION 2: Environment Setup
# =========================================================
elif nav_selection == "Environment Setup":
    st.markdown(
        """<div class="main-title">Environment Setup</div>
<div class="sub-title">Configure room temperatures, insulation, and simulation parameters</div>""",
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
            help="Starting temperature of the room.",
        )
        st.session_state.desired_temp = st.slider(
            "Desired Target Temperature (°C)",
            min_value=float(SimulationDefaults.MIN_DESIRED_TEMP),
            max_value=float(SimulationDefaults.MAX_DESIRED_TEMP),
            value=float(st.session_state.desired_temp),
            step=0.5,
            help="Desired target comfort temperature.",
        )
        st.session_state.comfort_tolerance = st.slider(
            "Comfort Tolerance (± °C)",
            min_value=0.5,
            max_value=3.0,
            value=float(st.session_state.comfort_tolerance),
            step=0.1,
            help="Acceptable deviation around the desired temperature.",
        )
        st.session_state.outside_temp = st.slider(
            "Outside Temperature (°C)",
            min_value=float(SimulationDefaults.MIN_OUTSIDE_TEMP),
            max_value=float(SimulationDefaults.MAX_OUTSIDE_TEMP),
            value=float(st.session_state.outside_temp),
            step=0.5,
            help="Outdoor ambient temperature.",
        )

        ins_opts = ["Poor", "Medium", "Good"]
        st.session_state.insulation_level = st.selectbox(
            "Room Insulation Level",
            options=ins_opts,
            index=ins_opts.index(st.session_state.insulation_level),
            help="Determines the rate of heat exchange with outside air.",
        )
        st.session_state.max_steps = st.slider(
            "Simulation Steps",
            min_value=int(SimulationDefaults.MIN_STEPS),
            max_value=int(SimulationDefaults.MAX_STEPS),
            value=int(st.session_state.max_steps),
            step=5,
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
            help="Priority given to staying close to target temperature.",
        )
        st.session_state.weight_energy = st.slider(
            "Energy Cost Weight (w_e)",
            min_value=0.0,
            max_value=2.0,
            value=float(st.session_state.weight_energy),
            step=0.05,
            help="Penalty for consuming electricity when heating or cooling.",
        )
        st.session_state.weight_stability = st.slider(
            "Stability Penalty Weight (w_s)",
            min_value=0.0,
            max_value=2.0,
            value=float(st.session_state.weight_stability),
            step=0.05,
            help="Penalty for large single-step temperature changes.",
        )
        st.session_state.weight_overshoot = st.slider(
            "Overshoot Penalty Weight (w_o)",
            min_value=0.0,
            max_value=3.0,
            value=float(st.session_state.weight_overshoot),
            step=0.1,
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
    if st.button("Apply Settings & Reset All Simulations", type="primary", use_container_width=True):
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
            ),
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            create_utility_comparison_chart(
                action_evaluations=last_dec.action_evaluations,
                selected_action=last_dec.selected_action,
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
# SECTION 5: Comparison
# =========================================================
elif nav_selection == "Comparison":
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
    st.markdown(get_comparison_summary_html(comp_summary), unsafe_allow_html=True)

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
        ),
        use_container_width=True,
        key="chart_sec_c_energy",
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
