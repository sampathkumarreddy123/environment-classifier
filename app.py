"""
Main Streamlit Application for the Environment Classifier and Demonstrator.

Features:
- Head-to-head comparison of Simple Reflex Thermostat vs. Utility-Based Agent
- Thermodynamic room simulation (Newton's cooling law, HVAC actuation, disturbance noise)
- Interactive parameter tuning (comfort weight, energy weight, insulation, outside temperature)
- Rich Plotly visualizations: gauges, synchronized comparative trajectories, action distributions, and utility breakdowns
"""

import streamlit as st
import pandas as pd

from config import (
    CLASS_COLD,
    CLASS_COMFORTABLE,
    CLASS_HOT,
    STATUS_COLORS,
    SimulationDefaults,
    AgentDefaults,
    ACTION_OFF,
    ACTION_HEAT,
    ACTION_COOL,
    AGENT_REFLEX,
    AGENT_UTILITY,
)
from environment import RoomEnvironment
from agents import SimpleReflexAgent, UtilityBasedAgent
from utils import (
    create_comparison_bar_chart,
    create_temperature_gauge,
    create_temperature_history_chart,
    create_dual_comparison_chart,
    create_action_distribution_chart,
    create_utility_breakdown_chart,
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
        margin-bottom: 0.5rem;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 1.15rem;
        font-weight: 500;
        color: #0284c7;
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
    }
    .scorecard {
        background: white;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
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
    """Initialize or reset application session state."""
    # Shared configuration defaults
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

    # Agent hyperparameters
    if "weight_comfort" not in st.session_state:
        st.session_state.weight_comfort = AgentDefaults.DEFAULT_WEIGHT_COMFORT
    if "weight_energy" not in st.session_state:
        st.session_state.weight_energy = AgentDefaults.DEFAULT_WEIGHT_ENERGY
    if "weight_switching" not in st.session_state:
        st.session_state.weight_switching = AgentDefaults.DEFAULT_WEIGHT_SWITCHING
    if "reflex_hysteresis" not in st.session_state:
        st.session_state.reflex_hysteresis = AgentDefaults.DEFAULT_REFLEX_HYSTERESIS

    # Single environment sandbox (for Tab 2)
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

    # Dual comparison environments and agents (for Tab 1)
    if "env_reflex" not in st.session_state:
        reset_dual_simulation()


def reset_dual_simulation() -> None:
    """Reset both comparison environments and agents to identical baseline states."""
    st.session_state.env_reflex = RoomEnvironment(
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
    st.session_state.env_utility = RoomEnvironment(
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
    st.session_state.agent_reflex = SimpleReflexAgent(
        hysteresis=st.session_state.reflex_hysteresis
    )
    st.session_state.agent_utility = UtilityBasedAgent(
        weight_comfort=st.session_state.weight_comfort,
        weight_energy=st.session_state.weight_energy,
        weight_switching=st.session_state.weight_switching,
    )


init_session_state()


# ---------------------------------------------------------
# Sidebar: Environment & Agent Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Simulation Settings")

    with st.expander("🌡️ Thermal Parameters", expanded=True):
        st.session_state.initial_temp = st.slider(
            "Initial Room Temperature (°C)",
            min_value=float(SimulationDefaults.MIN_INITIAL_TEMP),
            max_value=float(SimulationDefaults.MAX_INITIAL_TEMP),
            value=float(st.session_state.initial_temp),
            step=0.5,
        )
        st.session_state.desired_temp = st.slider(
            "Desired Comfort Target (°C)",
            min_value=float(SimulationDefaults.MIN_DESIRED_TEMP),
            max_value=float(SimulationDefaults.MAX_DESIRED_TEMP),
            value=float(st.session_state.desired_temp),
            step=0.5,
        )
        st.session_state.comfort_tolerance = st.slider(
            "Comfort Tolerance (± °C)",
            min_value=0.5,
            max_value=3.0,
            value=float(st.session_state.comfort_tolerance),
            step=0.1,
        )
        st.session_state.outside_temp = st.slider(
            "Outside Temperature (°C)",
            min_value=float(SimulationDefaults.MIN_OUTSIDE_TEMP),
            max_value=float(SimulationDefaults.MAX_OUTSIDE_TEMP),
            value=float(st.session_state.outside_temp),
            step=0.5,
        )

    with st.expander("🏠 Room & HVAC Physical Properties", expanded=False):
        ins_options = ["Poor", "Medium", "Good"]
        st.session_state.insulation_level = st.selectbox(
            "Room Insulation Level",
            options=ins_options,
            index=ins_options.index(st.session_state.insulation_level),
            help="Higher insulation reduces heat exchange with outdoor air.",
        )
        st.session_state.heater_power = st.slider(
            "Heater Power (°C / step)",
            min_value=float(SimulationDefaults.MIN_HEATER_POWER),
            max_value=float(SimulationDefaults.MAX_HEATER_POWER),
            value=float(st.session_state.heater_power),
            step=0.1,
        )
        st.session_state.cooling_power = st.slider(
            "Cooling Power (°C / step)",
            min_value=float(SimulationDefaults.MIN_COOLING_POWER),
            max_value=float(SimulationDefaults.MAX_COOLING_POWER),
            value=float(st.session_state.cooling_power),
            step=0.1,
        )
        st.session_state.disturbance_level = st.slider(
            "Disturbance Noise (Std Dev)",
            min_value=float(SimulationDefaults.MIN_DISTURBANCE),
            max_value=float(SimulationDefaults.MAX_DISTURBANCE),
            value=float(st.session_state.disturbance_level),
            step=0.05,
        )

    with st.expander("🤖 Agent Hyperparameters", expanded=True):
        st.caption("Utility-Based Agent Weights:")
        st.session_state.weight_comfort = st.slider(
            "Comfort Weight (w_c)",
            min_value=0.1,
            max_value=3.0,
            value=float(st.session_state.weight_comfort),
            step=0.1,
            help="Higher value prioritizes keeping temperature strictly in the target zone.",
        )
        st.session_state.weight_energy = st.slider(
            "Energy Cost Weight (w_e)",
            min_value=0.0,
            max_value=2.0,
            value=float(st.session_state.weight_energy),
            step=0.05,
            help="Higher value penalizes running the heater or cooler when passive drift is acceptable.",
        )
        st.session_state.weight_switching = st.slider(
            "Switching Penalty (w_s)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.weight_switching),
            step=0.05,
            help="Penalizes toggling between heating/cooling to reduce equipment wear.",
        )

        st.caption("Simple Reflex Agent Settings:")
        st.session_state.reflex_hysteresis = st.slider(
            "Reflex Deadband / Hysteresis (°C)",
            min_value=0.0,
            max_value=1.5,
            value=float(st.session_state.reflex_hysteresis),
            step=0.1,
            help="Additional tolerance offset before reflex thermostat triggers.",
        )

    with st.expander("⏱️ Simulation Steps & Reproducibility", expanded=False):
        st.session_state.max_steps = st.slider(
            "Simulation Steps",
            min_value=int(SimulationDefaults.MIN_STEPS),
            max_value=int(SimulationDefaults.MAX_STEPS),
            value=int(st.session_state.max_steps),
            step=5,
        )
        st.session_state.random_seed = st.number_input(
            "Random Seed",
            value=int(st.session_state.random_seed),
            step=1,
            help="Ensures identical disturbance noise across comparative runs.",
        )

    if st.button("🔄 Apply Config & Reset Simulation", use_container_width=True, type="primary"):
        reset_dual_simulation()
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
        st.rerun()


# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">Environment Classifier and Demonstrator</div>
        <div class="main-subtitle">Intelligent Room Temperature Control Simulation</div>
    </div>
    <div class="desc-box">
        Compare an <b>optimization-driven Utility-Based Agent</b> against a <b>threshold-driven Simple Reflex Thermostat</b>
        within a continuous thermodynamic room simulation. Both agents operate under <b>identical physical laws, outside weather,
        and random disturbance seeds</b> for a rigorous head-to-head benchmark.
    </div>
    """,
    unsafe_allow_html=True,
)

tab_comparison, tab_sandbox, tab_theory = st.tabs([
    "⚔️ Agent Head-to-Head Comparison",
    "🧪 Manual Room Sandbox",
    "📖 Thermodynamic & Agent Theory",
])


# =========================================================
# TAB 1: Agent Head-to-Head Comparison
# =========================================================
with tab_comparison:
    env_r: RoomEnvironment = st.session_state.env_reflex
    env_u: RoomEnvironment = st.session_state.env_utility
    agent_r: SimpleReflexAgent = st.session_state.agent_reflex
    agent_u: UtilityBasedAgent = st.session_state.agent_utility

    # Simulation Execution Bar
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1, 1, 1, 2])
    with col_ctrl1:
        if st.button("▶️ Step 1 Step", key="btn_step_dual", use_container_width=True):
            if env_r.current_step < st.session_state.max_steps:
                obs_r = env_r.get_state()
                act_r = agent_r.act(obs_r)
                env_r.step(act_r)

                obs_u = env_u.get_state()
                act_u = agent_u.act(obs_u)
                env_u.step(act_u)
                st.rerun()
            else:
                st.warning("Max steps reached. Click Reset to run again.")

    with col_ctrl2:
        if st.button("⚡ Run Full Simulation", key="btn_run_dual", use_container_width=True, type="primary"):
            remaining = st.session_state.max_steps - env_r.current_step
            if remaining > 0:
                for _ in range(remaining):
                    obs_r = env_r.get_state()
                    act_r = agent_r.act(obs_r)
                    env_r.step(act_r)

                    obs_u = env_u.get_state()
                    act_u = agent_u.act(obs_u)
                    env_u.step(act_u)
                st.rerun()
            else:
                st.warning("Max steps reached. Click Reset to run again.")

    with col_ctrl3:
        if st.button("↩️ Reset Comparison", key="btn_reset_dual", use_container_width=True):
            reset_dual_simulation()
            st.rerun()

    with col_ctrl4:
        progress_val = min(1.0, env_r.current_step / st.session_state.max_steps)
        st.progress(progress_val, text=f"Step {env_r.current_step} of {st.session_state.max_steps}")

    # Synchronized Comparative Trajectory Chart
    df_r = env_r.get_history_df()
    df_u = env_u.get_history_df()
    dual_fig = create_dual_comparison_chart(
        df_reflex=df_r,
        df_utility=df_u,
        desired_temp=st.session_state.desired_temp,
        comfort_tolerance=st.session_state.comfort_tolerance,
    )
    st.plotly_chart(dual_fig, use_container_width=True)

    # Side-by-Side Agent Scorecards
    stats_r = env_r.get_summary_stats()
    stats_u = env_u.get_summary_stats()
    metrics_r = agent_r.get_metrics()
    metrics_u = agent_u.get_metrics()

    st.subheader("📊 Performance Scorecard & Trade-Off Analysis")
    col_card_r, col_card_u = st.columns(2)

    with col_card_r:
        st.markdown(
            f"""
            <div style="border-top: 4px solid #dc2626; background: #fff5f5; border-radius: 8px; padding: 16px; border: 1px solid #fecaca;">
                <h4 style="margin: 0 0 10px 0; color: #b91c1c;">🤖 Simple Reflex Thermostat</h4>
                <div style="font-size: 13px; color: #475569; margin-bottom: 12px;"><b>Strategy:</b> Reactive fixed thresholds. Turns on full heat when below target; turns on full cooling when above target.</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                    <div><b>Comfort Time:</b> <span style="font-size: 16px; color: #1e293b; font-weight: bold;">{stats_r['comfort_percentage']}%</span></div>
                    <div><b>Total Energy:</b> <span style="font-size: 16px; color: #dc2626; font-weight: bold;">{metrics_r['total_energy']} units</span></div>
                    <div><b>Mean Absolute Error:</b> <span style="font-size: 16px; color: #1e293b; font-weight: bold;">{stats_r['avg_error']:.2f} °C</span></div>
                    <div><b>Actuator Switches:</b> <span style="font-size: 16px; color: #1e293b; font-weight: bold;">{metrics_r['switch_count']} toggles</span></div>
                    <div><b>Current Temp:</b> <span style="font-size: 16px; color: #1e293b; font-weight: bold;">{env_r.current_temp:.2f} °C</span></div>
                    <div><b>Last Action:</b> <span style="font-size: 16px; color: #dc2626; font-weight: bold;">{agent_r.last_action}</span></div>
                </div>
                <div style="margin-top: 10px; font-size: 12px; color: #64748b; font-style: italic;">{agent_r.last_reasoning}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_card_u:
        st.markdown(
            f"""
            <div style="border-top: 4px solid #2563eb; background: #eff6ff; border-radius: 8px; padding: 16px; border: 1px solid #bfdbfe;">
                <h4 style="margin: 0 0 10px 0; color: #1d4ed8;">🧠 Utility-Based Agent</h4>
                <div style="font-size: 13px; color: #475569; margin-bottom: 12px;"><b>Strategy:</b> Thermal outcome projection + multi-objective optimization (Comfort vs. Energy Cost vs. Switching).</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                    <div><b>Comfort Time:</b> <span style="font-size: 16px; color: #1e293b; font-weight: bold;">{stats_u['comfort_percentage']}%</span></div>
                    <div><b>Total Energy:</b> <span style="font-size: 16px; color: #2563eb; font-weight: bold;">{metrics_u['total_energy']} units</span></div>
                    <div><b>Mean Absolute Error:</b> <span style="font-size: 16px; color: #1e293b; font-weight: bold;">{stats_u['avg_error']:.2f} °C</span></div>
                    <div><b>Actuator Switches:</b> <span style="font-size: 16px; color: #1e293b; font-weight: bold;">{metrics_u['switch_count']} toggles</span></div>
                    <div><b>Current Temp:</b> <span style="font-size: 16px; color: #1e293b; font-weight: bold;">{env_u.current_temp:.2f} °C</span></div>
                    <div><b>Last Action:</b> <span style="font-size: 16px; color: #2563eb; font-weight: bold;">{agent_u.last_action}</span></div>
                </div>
                <div style="margin-top: 10px; font-size: 12px; color: #64748b; font-style: italic;">{agent_u.last_reasoning}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Comparison Visualizations
    col_v1, col_v2 = st.columns([1, 1])
    with col_v1:
        act_fig = create_action_distribution_chart(metrics_r, metrics_u)
        st.plotly_chart(act_fig, use_container_width=True)

    with col_v2:
        if agent_u.last_action_utilities:
            u_breakdown_fig = create_utility_breakdown_chart(agent_u.last_action_utilities)
            st.plotly_chart(u_breakdown_fig, use_container_width=True)
        else:
            st.info("Run or step the simulation to inspect the utility decomposition chart.")

    # Combined Telemetry Download
    with st.expander("📥 Export Combined Simulation Log"):
        comp_df = pd.DataFrame({
            "step": df_r["step"],
            "outside_temp": df_r["outside_temp"],
            "reflex_temp": df_r["room_temp"],
            "reflex_action": df_r["action"],
            "reflex_status": df_r["classification"],
            "utility_temp": df_u["room_temp"],
            "utility_action": df_u["action"],
            "utility_status": df_u["classification"],
            "target_temp": df_r["desired_temp"],
        })
        st.dataframe(comp_df.tail(15), use_container_width=True, hide_index=True)
        csv_bytes = comp_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Comparative CSV Log",
            data=csv_bytes,
            file_name="agent_comparison_log.csv",
            mime="text/csv",
        )


# =========================================================
# TAB 2: Interactive Room Sandbox (Phase 1 Baseline)
# =========================================================
with tab_sandbox:
    s_env: RoomEnvironment = st.session_state.sandbox_env
    s_state = s_env.get_state()
    s_history = s_env.get_history_df()
    s_stats = s_env.get_summary_stats()

    st.markdown(
        get_status_card_html(
            classification=s_state["classification"],
            current_temp=s_state["room_temp"],
            desired_temp=s_state["desired_temp"],
        ),
        unsafe_allow_html=True,
    )

    # Sandbox Manual Controls
    col_sb1, col_sb2, col_sb3 = st.columns([2, 1, 1])
    with col_sb1:
        sandbox_action = st.radio(
            "Manual HVAC Action:",
            options=[ACTION_OFF, ACTION_HEAT, ACTION_COOL],
            horizontal=True,
            format_func=lambda x: {
                ACTION_OFF: "⏸️ OFF (Natural Drift)",
                ACTION_HEAT: "🔥 HEAT (+Heater)",
                ACTION_COOL: "❄️ COOL (-Cooler)",
            }[x],
        )
    with col_sb2:
        if st.button("▶️ Step 1 Step", key="sb_step", use_container_width=True):
            s_env.step(sandbox_action)
            st.rerun()
    with col_sb3:
        if st.button("↩️ Reset Sandbox", key="sb_reset", use_container_width=True):
            s_env.reset()
            st.rerun()

    # Gauges & Bars
    row_g1, row_g2 = st.columns([1, 1])
    with row_g1:
        gauge_fig = create_temperature_gauge(
            current_temp=s_state["room_temp"],
            desired_temp=s_state["desired_temp"],
            comfort_tolerance=s_env.comfort_tolerance,
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

    with row_g2:
        bar_fig = create_comparison_bar_chart(
            room_temp=s_state["room_temp"],
            desired_temp=s_state["desired_temp"],
            outside_temp=s_state["outside_temp"],
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # History line chart
    s_history_fig = create_temperature_history_chart(
        df=s_history,
        desired_temp=s_state["desired_temp"],
        comfort_tolerance=s_env.comfort_tolerance,
    )
    st.plotly_chart(s_history_fig, use_container_width=True)


# =========================================================
# TAB 3: Thermodynamic & Agent Theory
# =========================================================
with tab_theory:
    st.markdown(
        """
        ### 🧮 Theoretical Foundation

        #### 1. Room Heat Transfer Model (Newton's Cooling Law)
        Room temperature changes at each discrete time step according to:
        $$\\Delta T = k_{\\text{loss}} \\cdot (T_{\\text{outside}} - T_t) + \\Delta T_{\\text{HVAC}} + \\epsilon_t$$
        - **Insulation Factor ($k_{\\text{loss}}$):**
          - `Poor`: $k = 0.14$ (rapid heat dissipation)
          - `Medium`: $k = 0.07$ (standard double-glazed walls)
          - `Good`: $k = 0.025$ (well-sealed thermal envelope)
        - **HVAC Contribution:**
          - `HEAT`: $+P_{\\text{heater}}$
          - `COOL`: $-P_{\\text{cooler}}$
          - `OFF`: $0.0$
        - **Disturbance Noise:** $\\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2)$

        ---

        #### 2. Simple Reflex Thermostat Agent
        Operates on deterministic condition-action rules:
        - If $T_t < T_{\\text{desired}} - \\tau$: Actuate `HEAT`
        - If $T_t > T_{\\text{desired}} + \\tau$: Actuate `COOL`
        - Else: Maintain `OFF`
        *Limitation:* Does not foresee ambient thermal inertia or weigh energy costs; frequently suffers from threshold overshoot and rapid cycling.

        ---

        #### 3. Utility-Based Agent
        Predicts future temperature outcomes $\\hat{T}_{t+1}(a)$ for each action $a \\in \\{\\text{OFF}, \\text{HEAT}, \\text{COOL}\\}$ using its internal model, then selects:
        $$a^* = \\arg\\max_{a} \\left[ U_{\\text{comfort}}(\\hat{T}_{t+1}(a)) + U_{\\text{energy}}(a) + U_{\\text{switch}}(a) \\right]$$

        - **Comfort Utility:**
          $$U_{\\text{comfort}} = -w_c \\cdot \\left(\\frac{|\\hat{T} - T_{\\text{desired}}|}{\\tau}\\right)^2 + \\text{Bonus}_{\\text{comfort}}$$
        - **Energy Penalty:**
          $$U_{\\text{energy}} = -w_e \\cdot \\text{Cost}(a)$$
        - **Switching Penalty:**
          $$U_{\\text{switch}} = -w_s \\cdot \\mathbf{1}_{\\{a \\ne a_{\\text{prev}}\\}}$$
        """
    )
