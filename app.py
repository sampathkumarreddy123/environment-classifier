"""
Main Streamlit Application for the Environment Classifier and Demonstrator.
Supports Phase 1 (Room Simulation & Sandbox), Phase 2 (Utility-Based Intelligent Agent),
and Phase 3 (Simple Reflex Thermostat Agent).
"""

from typing import Dict, Any
import streamlit as st
import pandas as pd

from config import (
    CLASS_COLD,
    CLASS_COMFORTABLE,
    CLASS_HOT,
    STATUS_COLORS,
    SimulationDefaults,
    UtilityWeightsDefaults,
    ReflexThermostatDefaults,
    ACTION_OFF,
    ACTION_HEAT,
    ACTION_COOL,
    ACTION_METADATA,
)
from environment import RoomEnvironment
from utility_agent import UtilityBasedAgent
from reflex_agent import SimpleReflexAgent
from utils import (
    create_agent_temperature_history_chart,
    create_comparison_bar_chart,
    create_reflex_action_summary_chart,
    create_reflex_temperature_history_chart,
    create_temperature_gauge,
    create_utility_comparison_chart,
    create_utility_factors_breakdown_chart,
    get_decision_card_html,
    get_reflex_decision_card_html,
    get_reflex_rules_html,
    get_status_card_html,
)

# Page configuration
st.set_page_config(
    page_title="Environment Classifier & Demonstrator",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-header {
        margin-bottom: 0.4rem;
    }
    .main-title {
        font-size: 2.15rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 1.15rem;
        font-weight: 600;
        color: #0284c7;
        margin-bottom: 0.4rem;
    }
    .desc-box {
        background-color: #f8fafc;
        border-left: 4px solid #0284c7;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 0.95rem;
        color: #334155;
        margin-bottom: 1.2rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    """Initialize application session state variables."""
    # Physical and thermodynamic parameters
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

    # Primary simulation instances (Phase 2 Utility Agent + Environment)
    if "env" not in st.session_state:
        reset_simulation()

    # Independent Simple Reflex simulation instances (Phase 3 Reflex Agent + Environment)
    if "reflex_env" not in st.session_state:
        reset_reflex_simulation()

    # Manual sandbox instance (Phase 1 feature preserved intact)
    if "sandbox_env" not in st.session_state:
        st.session_state.sandbox_env = RoomEnvironment(
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


def _create_configured_environment() -> RoomEnvironment:
    """Instantiate a RoomEnvironment using current sidebar settings."""
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


def reset_simulation() -> None:
    """Reinstantiate the environment and utility agent with current settings."""
    st.session_state.env = _create_configured_environment()
    st.session_state.agent = UtilityBasedAgent(
        weight_comfort=st.session_state.weight_comfort,
        weight_energy=st.session_state.weight_energy,
        weight_stability=st.session_state.weight_stability,
        weight_overshoot=st.session_state.weight_overshoot,
    )
    # Combined step telemetry log
    st.session_state.simulation_records = []

    # Run initial baseline step 0 evaluation
    init_obs = st.session_state.env.get_state()
    initial_decision = st.session_state.agent.decide_action(init_obs)
    # Undo step 0 energy tracking so it begins cleanly at step 1
    st.session_state.agent.total_energy_consumed = 0.0
    st.session_state.agent.action_history = []
    st.session_state.agent.decision_history = []

    st.session_state.simulation_records.append({
        "step": 0,
        "room_temp": init_obs["room_temp"],
        "outside_temp": init_obs["outside_temp"],
        "desired_temp": init_obs["desired_temp"],
        "action": ACTION_OFF,
        "predicted_temp": init_obs["room_temp"],
        "actual_temp": init_obs["room_temp"],
        "utility_score": initial_decision.utility_score,
        "energy_consumed": 0.0,
        "classification": init_obs["classification"],
    })


def reset_reflex_simulation() -> None:
    """Reinstantiate the independent environment and simple reflex thermostat."""
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


init_session_state()
env: RoomEnvironment = st.session_state.env
agent: UtilityBasedAgent = st.session_state.agent
reflex_env: RoomEnvironment = st.session_state.reflex_env
reflex_agent: SimpleReflexAgent = st.session_state.reflex_agent


# ---------------------------------------------------------
# Sidebar: Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Environment Configuration")

    with st.expander("🌡️ Temperature Targets", expanded=True):
        st.session_state.initial_temp = st.slider(
            "Initial Room Temperature (°C)",
            min_value=float(SimulationDefaults.MIN_INITIAL_TEMP),
            max_value=float(SimulationDefaults.MAX_INITIAL_TEMP),
            value=float(st.session_state.initial_temp),
            step=0.5,
            help="Starting indoor temperature.",
        )
        st.session_state.desired_temp = st.slider(
            "Desired Temperature (°C)",
            min_value=float(SimulationDefaults.MIN_DESIRED_TEMP),
            max_value=float(SimulationDefaults.MAX_DESIRED_TEMP),
            value=float(st.session_state.desired_temp),
            step=0.5,
            help="Configured acceptable comfort target.",
        )
        st.session_state.comfort_tolerance = st.slider(
            "Comfort Tolerance (± °C)",
            min_value=0.5,
            max_value=3.0,
            value=float(st.session_state.comfort_tolerance),
            step=0.1,
            help="Acceptable temperature delta around desired temperature.",
        )
        st.session_state.outside_temp = st.slider(
            "Outside Temperature (°C)",
            min_value=float(SimulationDefaults.MIN_OUTSIDE_TEMP),
            max_value=float(SimulationDefaults.MAX_OUTSIDE_TEMP),
            value=float(st.session_state.outside_temp),
            step=0.5,
            help="External ambient temperature.",
        )

    with st.expander("🏠 Physical Properties", expanded=False):
        ins_options = ["Poor", "Medium", "Good"]
        st.session_state.insulation_level = st.selectbox(
            "Room Insulation Level",
            options=ins_options,
            index=ins_options.index(st.session_state.insulation_level),
            help="Insulation quality governs passive heat loss.",
        )
        st.session_state.heater_power = st.slider(
            "Heater Power (°C / step)",
            min_value=float(SimulationDefaults.MIN_HEATER_POWER),
            max_value=float(SimulationDefaults.MAX_HEATER_POWER),
            value=float(st.session_state.heater_power),
            step=0.1,
            help="Temperature increase delivered when HEAT is selected.",
        )
        st.session_state.cooling_power = st.slider(
            "Cooling Power (°C / step)",
            min_value=float(SimulationDefaults.MIN_COOLING_POWER),
            max_value=float(SimulationDefaults.MAX_COOLING_POWER),
            value=float(st.session_state.cooling_power),
            step=0.1,
            help="Temperature reduction delivered when COOL is selected.",
        )
        st.session_state.disturbance_level = st.slider(
            "Temperature Disturbance Level",
            min_value=float(SimulationDefaults.MIN_DISTURBANCE),
            max_value=float(SimulationDefaults.MAX_DISTURBANCE),
            value=float(st.session_state.disturbance_level),
            step=0.05,
            help="Standard deviation of random Gaussian environmental noise.",
        )

    with st.expander("⚖️ Utility Function Weights", expanded=True):
        st.markdown(
            "$$\\text{Utility} = - w_c P_{\\text{comfort}} - w_e C_{\\text{energy}} - w_s P_{\\text{instability}} - w_o P_{\\text{overshoot}}$$"
        )
        st.session_state.weight_comfort = st.slider(
            "Comfort Penalty Weight (w_c)",
            min_value=0.0,
            max_value=4.0,
            value=float(st.session_state.weight_comfort),
            step=0.1,
            help="Importance of proximity to desired target temperature.",
        )
        st.session_state.weight_energy = st.slider(
            "Energy Cost Weight (w_e)",
            min_value=0.0,
            max_value=3.0,
            value=float(st.session_state.weight_energy),
            step=0.05,
            help="Penalizes power consumed by heating and cooling.",
        )
        st.session_state.weight_stability = st.slider(
            "Instability Penalty Weight (w_s)",
            min_value=0.0,
            max_value=2.0,
            value=float(st.session_state.weight_stability),
            step=0.05,
            help="Avoids large abrupt temperature fluctuations.",
        )
        st.session_state.weight_overshoot = st.slider(
            "Overshoot Penalty Weight (w_o)",
            min_value=0.0,
            max_value=3.0,
            value=float(st.session_state.weight_overshoot),
            step=0.1,
            help="Heavily penalizes exceeding the acceptable comfort boundaries.",
        )
        agent.update_weights(
            weight_comfort=st.session_state.weight_comfort,
            weight_energy=st.session_state.weight_energy,
            weight_stability=st.session_state.weight_stability,
            weight_overshoot=st.session_state.weight_overshoot,
        )

    def _on_reflex_lower_change() -> None:
        if st.session_state.reflex_lower_slider >= st.session_state.reflex_upper_slider:
            st.session_state.reflex_upper_slider = min(
                float(ReflexThermostatDefaults.MAX_THRESHOLD),
                round(
                    float(st.session_state.reflex_lower_slider)
                    + float(ReflexThermostatDefaults.THRESHOLD_STEP),
                    1,
                ),
            )

    def _on_reflex_upper_change() -> None:
        if st.session_state.reflex_upper_slider <= st.session_state.reflex_lower_slider:
            st.session_state.reflex_lower_slider = max(
                float(ReflexThermostatDefaults.MIN_THRESHOLD),
                round(
                    float(st.session_state.reflex_upper_slider)
                    - float(ReflexThermostatDefaults.THRESHOLD_STEP),
                    1,
                ),
            )

    with st.expander("🔁 Reflex Thermostat Thresholds", expanded=True):
        st.slider(
            "Lower Temperature Threshold (°C)",
            min_value=float(ReflexThermostatDefaults.MIN_THRESHOLD),
            max_value=float(
                ReflexThermostatDefaults.MAX_THRESHOLD - ReflexThermostatDefaults.THRESHOLD_STEP
            ),
            step=float(ReflexThermostatDefaults.THRESHOLD_STEP),
            key="reflex_lower_slider",
            on_change=_on_reflex_lower_change,
            help="Room temperature below this threshold triggers HEAT.",
        )
        st.slider(
            "Upper Temperature Threshold (°C)",
            min_value=float(
                ReflexThermostatDefaults.MIN_THRESHOLD + ReflexThermostatDefaults.THRESHOLD_STEP
            ),
            max_value=float(ReflexThermostatDefaults.MAX_THRESHOLD),
            step=float(ReflexThermostatDefaults.THRESHOLD_STEP),
            key="reflex_upper_slider",
            on_change=_on_reflex_upper_change,
            help="Room temperature above this threshold triggers COOL.",
        )
        st.caption(
            f"OFF Deadband: {st.session_state.reflex_lower_slider:.1f}°C – "
            f"{st.session_state.reflex_upper_slider:.1f}°C"
        )
        reflex_agent.update_thresholds(
            lower_threshold=st.session_state.reflex_lower_slider,
            upper_threshold=st.session_state.reflex_upper_slider,
        )

    with st.expander("⏱️ Horizon & Reproducibility", expanded=False):
        st.session_state.max_steps = st.slider(
            "Number of Simulation Steps",
            min_value=int(SimulationDefaults.MIN_STEPS),
            max_value=int(SimulationDefaults.MAX_STEPS),
            value=int(st.session_state.max_steps),
            step=5,
        )
        st.session_state.random_seed = st.number_input(
            "Random Disturbance Seed",
            value=int(st.session_state.random_seed),
            step=1,
            help="Ensures identical disturbance sequence across comparative runs.",
        )

    if st.button("🔄 Apply Configuration & Reset", use_container_width=True, type="primary"):
        reset_simulation()
        reset_reflex_simulation()
        st.rerun()


# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.markdown(
    """<div class="main-header">
<div class="main-title">Environment Classifier and Demonstrator</div>
<div class="main-subtitle">Intelligent Room Temperature Control Simulation</div>
</div>
<div class="desc-box">
<b>Independent Agent Environments:</b> Explore the <b>Utility-Based Intelligent Agent</b> (Phase 2, optimization-driven
over projected thermal states) alongside the <b>Simple Reflex Thermostat</b> (Phase 3, pure condition-action threshold rules),
plus the Phase 1 interactive room sandbox.
</div>""",
    unsafe_allow_html=True,
)

tab_agent, tab_reflex, tab_sandbox, tab_theory = st.tabs([
    "🤖 Utility-Based Agent Simulation",
    "🔁 Simple Reflex Thermostat",
    "🧪 Interactive Room Sandbox",
    "📖 Mathematical Model & Theory",
])


# =========================================================
# TAB 1: Utility-Based Agent Simulation
# =========================================================
with tab_agent:
    current_state = env.get_state()
    last_dec = agent.last_decision

    # 1. Main Environment Status Banner
    st.markdown(
        get_status_card_html(
            classification=current_state["classification"],
            current_temp=current_state["room_temp"],
            desired_temp=current_state["desired_temp"],
        ),
        unsafe_allow_html=True,
    )

    # 2. Main Dashboard Metrics
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
            label="Desired Target",
            value=f"{current_state['desired_temp']:.1f} °C",
            help=f"Acceptable range: {current_state['desired_temp'] - env.comfort_tolerance:.1f}°C to {current_state['desired_temp'] + env.comfort_tolerance:.1f}°C",
        )
    with m3:
        st.metric(
            label="Outside Temperature",
            value=f"{current_state['outside_temp']:.1f} °C",
            delta=f"{current_state['outside_temp'] - current_state['room_temp']:+.1f} °C ambient diff",
            delta_color="off",
        )
    with m4:
        st.metric(
            label="Environment Status",
            value=f"{current_state['classification']}",
            help="Cold, Comfortable, or Hot",
        )
    with m5:
        progress_pct = min(100.0, (env.current_step / st.session_state.max_steps) * 100.0)
        st.metric(
            label="Simulation Progress",
            value=f"Step {env.current_step} / {st.session_state.max_steps}",
            delta=f"{progress_pct:.0f}% completed",
            delta_color="normal",
        )

    # Progress bar
    st.progress(progress_pct / 100.0)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Interactive Simulation Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])

    def perform_agent_step() -> None:
        """Execute one complete observation-decision-action step."""
        if env.current_step < st.session_state.max_steps:
            obs = env.get_state()
            decision = agent.decide_action(obs)
            next_state = env.step(decision.selected_action)

            st.session_state.simulation_records.append({
                "step": env.current_step,
                "room_temp": obs["room_temp"],
                "outside_temp": obs["outside_temp"],
                "desired_temp": obs["desired_temp"],
                "action": decision.selected_action,
                "predicted_temp": decision.predicted_temp,
                "actual_temp": next_state["room_temp"],
                "utility_score": decision.utility_score,
                "energy_consumed": decision.energy_consumed,
                "classification": next_state["classification"],
            })

    with ctrl_col1:
        if st.button("▶️ Step 1 Step", key="btn_step_agent", use_container_width=True):
            if env.current_step < st.session_state.max_steps:
                perform_agent_step()
                st.rerun()
            else:
                st.warning("Maximum simulation steps reached. Click Reset to start again.")

    with ctrl_col2:
        if st.button("⚡ Run Full Simulation", key="btn_run_agent", use_container_width=True, type="primary"):
            remaining = st.session_state.max_steps - env.current_step
            if remaining > 0:
                for _ in range(remaining):
                    perform_agent_step()
                st.rerun()
            else:
                st.warning("Maximum simulation steps reached. Click Reset to start again.")

    with ctrl_col3:
        if st.button("↩️ Reset Simulation", key="btn_reset_agent", use_container_width=True):
            reset_simulation()
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Dedicated Utility-Based Agent Section
    st.subheader("🧠 Utility-Based Agent Decision Intelligence")

    if last_dec:
        # Decision Explanation Card
        decision_card_html = get_decision_card_html(
            current_temp=last_dec.current_temp,
            desired_temp=last_dec.desired_temp,
            selected_action=last_dec.selected_action,
            predicted_temp=last_dec.predicted_temp,
            utility_score=last_dec.utility_score,
            reason=last_dec.reason,
        )
        st.markdown(decision_card_html, unsafe_allow_html=True)

        # Agent Key Metrics Row
        ak1, ak2, ak3, ak4 = st.columns(4)
        with ak1:
            st.metric("Selected Action", f"{last_dec.selected_action}")
        with ak2:
            st.metric("Predicted Next Temp", f"{last_dec.predicted_temp:.2f} °C")
        with ak3:
            st.metric("Current Utility Score", f"{last_dec.utility_score:.2f}")
        with ak4:
            st.metric("Cumulative Energy", f"{agent.total_energy_consumed:.1f} units")

        # Visualizations: Utility Comparison & Penalty Breakdown
        u_col1, u_col2 = st.columns(2)
        with u_col1:
            fig_u_comp = create_utility_comparison_chart(
                action_evaluations=last_dec.action_evaluations,
                selected_action=last_dec.selected_action,
            )
            st.plotly_chart(fig_u_comp, use_container_width=True, key="chart_agent_utility_comparison")

        with u_col2:
            fig_u_factors = create_utility_factors_breakdown_chart(
                action_evaluations=last_dec.action_evaluations,
            )
            st.plotly_chart(fig_u_factors, use_container_width=True, key="chart_agent_utility_factors")

    # 5. Temperature Trajectory History Chart
    st.subheader("📈 Temperature Trajectory & Performance History")
    sim_df = pd.DataFrame(st.session_state.simulation_records)
    fig_history = create_agent_temperature_history_chart(
        df=sim_df,
        desired_temp=st.session_state.desired_temp,
        comfort_tolerance=st.session_state.comfort_tolerance,
    )
    st.plotly_chart(fig_history, use_container_width=True, key="chart_agent_temperature_history")

    # Gauges & Ambient Comparison
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        fig_gauge = create_temperature_gauge(
            current_temp=current_state["room_temp"],
            desired_temp=current_state["desired_temp"],
            comfort_tolerance=env.comfort_tolerance,
        )
        st.plotly_chart(fig_gauge, use_container_width=True, key="chart_agent_temperature_gauge")

    with g_col2:
        fig_bar = create_comparison_bar_chart(
            room_temp=current_state["room_temp"],
            desired_temp=current_state["desired_temp"],
            outside_temp=current_state["outside_temp"],
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="chart_agent_temperature_bar")

    # 6. Simulation Diagnostics & Data Table
    st.subheader("📋 Step-by-Step Simulation Telemetry")
    if not sim_df.empty:
        diag1, diag2, diag3, diag4 = st.columns(4)
        comfort_pct = (sim_df["classification"] == CLASS_COMFORTABLE).mean() * 100.0
        avg_err = (sim_df["actual_temp"] - sim_df["desired_temp"]).abs().mean()
        with diag1:
            st.metric("Comfort Zone Ratio", f"{comfort_pct:.1f}%")
        with diag2:
            st.metric("Mean Absolute Error", f"{avg_err:.2f} °C")
        with diag3:
            st.metric("Total Energy Used", f"{agent.total_energy_consumed:.1f} units")
        with diag4:
            st.metric("Total Steps", f"{len(sim_df) - 1}")

        st.dataframe(
            sim_df.tail(12),
            use_container_width=True,
            hide_index=True,
        )

        csv_data = sim_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Complete Simulation Telemetry (CSV)",
            data=csv_data,
            file_name="utility_agent_simulation_telemetry.csv",
            mime="text/csv",
        )


# =========================================================
# TAB 2: Simple Reflex Thermostat (Phase 3)
# =========================================================
with tab_reflex:
    reflex_state = reflex_env.get_state()
    reflex_preview = reflex_agent.preview_decision(reflex_state)
    last_reflex_dec = reflex_agent.last_decision

    # 1. Environment Status Banner
    st.markdown(
        get_status_card_html(
            classification=reflex_state["classification"],
            current_temp=reflex_state["room_temp"],
            desired_temp=reflex_state["desired_temp"],
        ),
        unsafe_allow_html=True,
    )

    # 2. Required Dashboard Metrics (Current Temp, Thresholds, Action, Classification, Energy)
    rm1, rm2, rm3, rm4, rm5, rm6 = st.columns(6)
    with rm1:
        st.metric(
            label="Current Room Temp",
            value=f"{reflex_state['room_temp']:.2f} °C",
            delta=f"{reflex_state['room_temp'] - reflex_state['desired_temp']:+.2f} °C vs Target",
            delta_color="inverse",
        )
    with rm2:
        st.metric(
            label="Lower Threshold",
            value=f"{reflex_agent.lower_threshold:.1f} °C",
            help="Triggers HEAT when room temperature drops below this value.",
        )
    with rm3:
        st.metric(
            label="Upper Threshold",
            value=f"{reflex_agent.upper_threshold:.1f} °C",
            help="Triggers COOL when room temperature rises above this value.",
        )
    with rm4:
        st.metric(
            label="Selected Action",
            value=f"{reflex_preview.selected_action}",
            help=f"Active rule: {reflex_preview.rule_triggered}",
        )
    with rm5:
        st.metric(
            label="Classification",
            value=f"{reflex_state['classification']}",
            help="Environment comfort classification (Cold, Comfortable, or Hot).",
        )
    with rm6:
        last_energy = last_reflex_dec.energy_consumed if last_reflex_dec else 0.0
        st.metric(
            label="Energy Consumption",
            value=f"{reflex_agent.total_energy_consumed:.1f} units",
            delta=f"+{last_energy:.1f} last step" if reflex_env.current_step > 0 else "0.0 initial",
            delta_color="off",
        )

    reflex_progress = min(100.0, (reflex_env.current_step / st.session_state.max_steps) * 100.0)
    st.caption(
        f"Simulation Progress: Step {reflex_env.current_step} / {st.session_state.max_steps} "
        f"({reflex_progress:.0f}% completed) · Outside Temp: {reflex_state['outside_temp']:.1f}°C · "
        f"Insulation: {reflex_state['insulation_level']}"
    )
    st.progress(reflex_progress / 100.0)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Independent Interactive Simulation Controls
    r_ctrl1, r_ctrl2, r_ctrl3 = st.columns([1, 1, 1])

    def perform_reflex_step() -> None:
        """Execute one observation-rule-action step for the Simple Reflex Thermostat."""
        if reflex_env.current_step < st.session_state.max_steps:
            record = reflex_agent.step_environment(reflex_env)
            st.session_state.reflex_simulation_records.append(record)

    with r_ctrl1:
        if st.button("▶️ Step 1 Step", key="btn_step_reflex", use_container_width=True):
            if reflex_env.current_step < st.session_state.max_steps:
                perform_reflex_step()
                st.rerun()
            else:
                st.warning("Maximum simulation steps reached. Click Reset to start again.")

    with r_ctrl2:
        if st.button("⚡ Run Full Simulation", key="btn_run_reflex", use_container_width=True, type="primary"):
            remaining = st.session_state.max_steps - reflex_env.current_step
            if remaining > 0:
                for _ in range(remaining):
                    perform_reflex_step()
                st.rerun()
            else:
                st.warning("Maximum simulation steps reached. Click Reset to start again.")

    with r_ctrl3:
        if st.button("↩️ Reset Simulation", key="btn_reset_reflex", use_container_width=True):
            reset_reflex_simulation()
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Visual Condition-Action Rules & Decision Card
    st.subheader("🔁 Simple Reflex Condition-Action Rules")
    st.markdown(
        get_reflex_rules_html(
            current_temp=reflex_state["room_temp"],
            lower_threshold=reflex_agent.lower_threshold,
            upper_threshold=reflex_agent.upper_threshold,
            selected_action=reflex_preview.selected_action,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        get_reflex_decision_card_html(
            current_temp=reflex_state["room_temp"],
            lower_threshold=reflex_agent.lower_threshold,
            upper_threshold=reflex_agent.upper_threshold,
            selected_action=reflex_preview.selected_action,
            classification=reflex_state["classification"],
            step_energy=last_reflex_dec.energy_consumed if last_reflex_dec else 0.0,
            total_energy=reflex_agent.total_energy_consumed,
            rule_triggered=reflex_preview.rule_triggered,
            reason=reflex_preview.reason,
            last_step_action=last_reflex_dec.selected_action if last_reflex_dec else None,
            last_step_num=reflex_env.current_step,
        ),
        unsafe_allow_html=True,
    )

    # 5. Temperature Trajectory & Gauges
    st.subheader("📈 Reflex Thermostat Trajectory & Threshold Tracking")
    reflex_df = pd.DataFrame(st.session_state.reflex_simulation_records)
    fig_reflex_history = create_reflex_temperature_history_chart(
        df=reflex_df,
        lower_threshold=reflex_agent.lower_threshold,
        upper_threshold=reflex_agent.upper_threshold,
        desired_temp=st.session_state.desired_temp,
    )
    st.plotly_chart(fig_reflex_history, use_container_width=True, key="chart_reflex_temperature_history")

    rg_col1, rg_col2 = st.columns(2)
    with rg_col1:
        st.plotly_chart(
            create_temperature_gauge(
                current_temp=reflex_state["room_temp"],
                desired_temp=reflex_state["desired_temp"],
                comfort_tolerance=reflex_env.comfort_tolerance,
            ),
            use_container_width=True,
            key="chart_reflex_temperature_gauge",
        )
    with rg_col2:
        st.plotly_chart(
            create_reflex_action_summary_chart(df=reflex_df),
            use_container_width=True,
            key="chart_reflex_action_summary",
        )

    # 6. Step-by-Step Simulation Telemetry (Pandas DataFrame)
    st.subheader("📋 Step-by-Step Reflex Simulation Telemetry")
    if not reflex_df.empty:
        r_diag1, r_diag2, r_diag3, r_diag4 = st.columns(4)
        reflex_comfort_pct = (reflex_df["classification"] == CLASS_COMFORTABLE).mean() * 100.0
        reflex_avg_err = (reflex_df["actual_temp"] - reflex_df["desired_temp"]).abs().mean()
        with r_diag1:
            st.metric("Comfort Zone Ratio", f"{reflex_comfort_pct:.1f}%")
        with r_diag2:
            st.metric("Mean Absolute Error", f"{reflex_avg_err:.2f} °C")
        with r_diag3:
            st.metric("Total Energy Used", f"{reflex_agent.total_energy_consumed:.1f} units")
        with r_diag4:
            st.metric("Total Steps", f"{len(reflex_df) - 1}")

        st.dataframe(
            reflex_df.tail(12),
            use_container_width=True,
            hide_index=True,
        )

        reflex_csv = reflex_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Reflex Simulation Telemetry (CSV)",
            data=reflex_csv,
            file_name="simple_reflex_simulation_telemetry.csv",
            mime="text/csv",
            key="btn_download_reflex_csv",
        )


# =========================================================
# TAB 3: Interactive Room Sandbox (Phase 1 Preserved)
# =========================================================
with tab_sandbox:
    s_env: RoomEnvironment = st.session_state.sandbox_env
    s_state = s_env.get_state()
    s_history = s_env.get_history_df()

    st.markdown("### 🧪 Manual Thermodynamic Room Explorer")
    st.caption("Inspect passive thermal drift and test manual HVAC commands without the agent.")

    st.markdown(
        get_status_card_html(
            classification=s_state["classification"],
            current_temp=s_state["room_temp"],
            desired_temp=s_state["desired_temp"],
        ),
        unsafe_allow_html=True,
    )

    sb_c1, sb_c2, sb_c3 = st.columns([2, 1, 1])
    with sb_c1:
        manual_act = st.radio(
            "Manual HVAC Actuation:",
            options=[ACTION_OFF, ACTION_HEAT, ACTION_COOL],
            horizontal=True,
            format_func=lambda x: {
                ACTION_OFF: "⏸️ OFF (Passive Drift)",
                ACTION_HEAT: "🔥 HEAT (+Heater)",
                ACTION_COOL: "❄️ COOL (-Cooler)",
            }[x],
        )
    with sb_c2:
        if st.button("▶️ Step Sandbox", key="sb_step_btn", use_container_width=True):
            s_env.step(manual_act)
            st.rerun()
    with sb_c3:
        if st.button("↩️ Reset Sandbox", key="sb_reset_btn", use_container_width=True):
            s_env.reset()
            st.rerun()

    sb_g1, sb_g2 = st.columns(2)
    with sb_g1:
        st.plotly_chart(
            create_temperature_gauge(
                current_temp=s_state["room_temp"],
                desired_temp=s_state["desired_temp"],
                comfort_tolerance=s_env.comfort_tolerance,
            ),
            use_container_width=True,
            key="chart_sandbox_temperature_gauge",
        )
    with sb_g2:
        st.plotly_chart(
            create_comparison_bar_chart(
                room_temp=s_state["room_temp"],
                desired_temp=s_state["desired_temp"],
                outside_temp=s_state["outside_temp"],
            ),
            use_container_width=True,
            key="chart_sandbox_temperature_bar",
        )


# =========================================================
# TAB 4: Mathematical Model & Theory
# =========================================================
with tab_theory:
    st.markdown(
        """
        ### 🧮 Mathematical Model & Decision Theory

        #### 1. Predictive Thermodynamic Model
        The room environment updates indoor temperature $T_{t+1}$ for each action $a \\in \\{\\text{HEAT}, \\text{COOL}, \\text{OFF}\\}$ using:
        $$T_{t+1} = T_t + k_{\\text{loss}} \\cdot (T_{\\text{outside}} - T_t) + \\Delta T_{\\text{HVAC}}(a) + \\epsilon_t$$
        Where:
        - $k_{\\text{loss}}$ is the building envelope insulation coefficient (`Poor`: $0.14$, `Medium`: $0.07$, `Good`: $0.025$).
        - $\\Delta T_{\\text{HVAC}}(a)$ is $+P_{\\text{heater}}$ for `HEAT`, $-P_{\\text{cooling}}$ for `COOL`, and $0$ for `OFF`.
        - $\\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2)$ is optional environmental disturbance noise.

        ---

        #### 2. Phase 2: Utility-Based Intelligent Agent
        The utility-based agent scores each candidate action using predicted temperature $\\hat{T}_{t+1}(a)$ and a weighted penalty formulation:
        $$\\text{Utility}(a) = - \\left( w_c \\cdot P_{\\text{comfort}} + w_e \\cdot C_{\\text{energy}} + w_s \\cdot P_{\\text{instability}} + w_o \\cdot P_{\\text{overshoot}} \\right)$$

        1. **Comfort Penalty ($P_{\\text{comfort}}$):** Proximity to the desired temperature target:
           $$P_{\\text{comfort}} = |\\hat{T}_{t+1}(a) - T_{\\text{desired}}|$$

        2. **Energy Cost ($C_{\\text{energy}}$):** Power consumed by active actuation:
           $$C_{\\text{energy}}(\\text{HEAT}) = 1.0, \\quad C_{\\text{energy}}(\\text{COOL}) = 1.0, \\quad C_{\\text{energy}}(\\text{OFF}) = 0.0$$

        3. **Temperature Instability Penalty ($P_{\\text{instability}}$):** Penalizes rapid abrupt swings:
           $$P_{\\text{instability}} = |\\hat{T}_{t+1}(a) - T_t|$$

        4. **Comfort Overshoot Penalty ($P_{\\text{overshoot}}$):** Distance exceeding the acceptable comfort boundaries:
           $$P_{\\text{overshoot}} = \\max\\left(0, (T_{\\text{desired}} - \\tau) - \\hat{T}\\right) + \\max\\left(0, \\hat{T} - (T_{\\text{desired}} + \\tau)\\right)$$

        The utility-based agent selects the action with the maximum utility value:
        $$a^* = \\arg\\max_{a \\in \\{\\text{HEAT}, \\text{COOL}, \\text{OFF}\\}} \\text{Utility}(a)$$

        ---

        #### 3. Phase 3: Simple Reflex Thermostat (Condition-Action Rules)
        Unlike the utility-based agent, the simple reflex thermostat computes no utility scores and predicts no future temperatures. It acts purely on the current observed room temperature $T_t$ using fixed thresholds $T_{\\text{lower}} < T_{\\text{upper}}$:

        $$
        a_t = \\begin{cases}
        \\text{HEAT}, & \\text{if } T_t < T_{\\text{lower}} \\\\
        \\text{COOL}, & \\text{if } T_t > T_{\\text{upper}} \\\\
        \\text{OFF}, & \\text{if } T_{\\text{lower}} \\le T_t \\le T_{\\text{upper}}
        \\end{cases}
        $$
        """
    )
