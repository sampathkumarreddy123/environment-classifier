"""
Synchronized Multi-Agent Comparison Engine & Performance Evaluator (Phase 4).

Runs the Utility-Based Intelligent Agent and the Simple Reflex Thermostat under
identical environmental conditions and identical step-by-step disturbance sequences,
maintaining independent simulation states and computing objective performance metrics.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config import (
    ACTION_COOL,
    ACTION_HEAT,
    ACTION_OFF,
    CLASS_COLD,
    CLASS_COMFORTABLE,
    CLASS_HOT,
    ReflexThermostatDefaults,
    SimulationDefaults,
    UtilityWeightsDefaults,
)
from environment import RoomEnvironment
from reflex_agent import SimpleReflexAgent
from utility_agent import UtilityBasedAgent


AGENT_UTILITY = "Utility-Based Agent"
AGENT_REFLEX = "Simple Reflex Agent"
TIE_LABEL = "Tie"
METRIC_TOLERANCE = 1e-4


@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration snapshot shared across a synchronized comparison run."""

    initial_temp: float = SimulationDefaults.DEFAULT_INITIAL_TEMP
    desired_temp: float = SimulationDefaults.DEFAULT_DESIRED_TEMP
    outside_temp: float = SimulationDefaults.DEFAULT_OUTSIDE_TEMP
    insulation_level: str = "Medium"
    comfort_tolerance: float = SimulationDefaults.DEFAULT_COMFORT_TOLERANCE
    heater_power: float = SimulationDefaults.DEFAULT_HEATER_POWER
    cooling_power: float = SimulationDefaults.DEFAULT_COOLING_POWER
    disturbance_level: float = SimulationDefaults.DEFAULT_DISTURBANCE
    max_steps: int = SimulationDefaults.DEFAULT_STEPS
    random_seed: Optional[int] = 42
    weight_comfort: float = UtilityWeightsDefaults.DEFAULT_WEIGHT_COMFORT
    weight_energy: float = UtilityWeightsDefaults.DEFAULT_WEIGHT_ENERGY
    weight_stability: float = UtilityWeightsDefaults.DEFAULT_WEIGHT_STABILITY
    weight_overshoot: float = UtilityWeightsDefaults.DEFAULT_WEIGHT_OVERSHOOT
    lower_threshold: float = ReflexThermostatDefaults.DEFAULT_LOWER_THRESHOLD
    upper_threshold: float = ReflexThermostatDefaults.DEFAULT_UPPER_THRESHOLD


@dataclass
class AgentPerformanceMetrics:
    """Quantitative temperature control, energy, stability, and classification metrics."""

    total_steps: int
    # Temperature Control
    avg_abs_error: float
    final_temp_error: float
    comfort_percentage: float
    comfort_steps: int
    overshoot_count: int
    # Energy Efficiency
    total_energy: float
    avg_energy_per_step: float
    heating_activations: int
    cooling_activations: int
    off_steps: int
    # Stability
    temp_fluctuations: float
    unnecessary_action_changes: int
    settling_steps: Optional[int]
    settling_label: str
    # Environment Classification Summary
    cold_steps: int
    comfortable_steps: int
    hot_steps: int
    cold_percentage: float
    comfortable_percentage: float
    hot_percentage: float


def compute_agent_metrics(df: pd.DataFrame) -> AgentPerformanceMetrics:
    """
    Calculate all performance and classification metrics from recorded simulation data.
    Handles zero-step and zero-energy edge cases safely.
    """
    if df.empty:
        return AgentPerformanceMetrics(
            total_steps=0,
            avg_abs_error=0.0,
            final_temp_error=0.0,
            comfort_percentage=0.0,
            comfort_steps=0,
            overshoot_count=0,
            total_energy=0.0,
            avg_energy_per_step=0.0,
            heating_activations=0,
            cooling_activations=0,
            off_steps=0,
            temp_fluctuations=0.0,
            unnecessary_action_changes=0,
            settling_steps=None,
            settling_label="Not settled (0 steps)",
            cold_steps=0,
            comfortable_steps=0,
            hot_steps=0,
            cold_percentage=0.0,
            comfortable_percentage=0.0,
            hot_percentage=0.0,
        )

    active_df = df[df["step"] > 0]
    has_active = not active_df.empty
    eval_df = active_df if has_active else df
    n_steps = len(active_df)

    # Temperature Control
    errors = (eval_df["actual_temp"] - eval_df["desired_temp"]).abs()
    avg_abs_error = round(float(errors.mean()), 3)
    final_temp_error = round(
        float(abs(df["actual_temp"].iloc[-1] - df["desired_temp"].iloc[-1])), 3
    )

    # Environment Classification counts & percentages
    cold_steps = int((eval_df["classification"] == CLASS_COLD).sum())
    comfortable_steps = int((eval_df["classification"] == CLASS_COMFORTABLE).sum())
    hot_steps = int((eval_df["classification"] == CLASS_HOT).sum())
    denom = len(eval_df)
    cold_pct = round((cold_steps / denom) * 100.0, 1) if denom > 0 else 0.0
    comfy_pct = round((comfortable_steps / denom) * 100.0, 1) if denom > 0 else 0.0
    hot_pct = round((hot_steps / denom) * 100.0, 1) if denom > 0 else 0.0

    # Overshoots: comfort boundary crossings across consecutive simulation steps
    overshoots = 0
    temps = df["actual_temp"].tolist()
    lowers = df["lower_comfort"].tolist()
    uppers = df["upper_comfort"].tolist()
    for i in range(1, len(temps)):
        prev_t, curr_t = temps[i - 1], temps[i]
        lower_c, upper_c = lowers[i], uppers[i]
        if (prev_t <= upper_c and curr_t > upper_c) or (
            prev_t >= lower_c and curr_t < lower_c
        ):
            overshoots += 1

    # Energy Efficiency (evaluated across active steps 1..N; 0.0 before step 1)
    if has_active:
        total_energy = round(float(active_df["energy_consumed"].sum()), 2)
        avg_energy_per_step = round(total_energy / n_steps, 3)
        heating_activations = int((active_df["action"] == ACTION_HEAT).sum())
        cooling_activations = int((active_df["action"] == ACTION_COOL).sum())
        off_steps = int((active_df["action"] == ACTION_OFF).sum())
    else:
        total_energy = 0.0
        avg_energy_per_step = 0.0
        heating_activations = 0
        cooling_activations = 0
        off_steps = 0

    # Stability: step-to-step temperature fluctuations (°C/step)
    if len(temps) > 1:
        step_deltas = [abs(temps[i] - temps[i - 1]) for i in range(1, len(temps))]
        temp_fluctuations = round(float(sum(step_deltas) / len(step_deltas)), 3)
    else:
        temp_fluctuations = 0.0

    # Stability: unnecessary action changes (direct HEAT<->COOL reversals or A->B->A chattering)
    unnecessary_changes = 0
    if has_active:
        actions = active_df["action"].tolist()
        for i in range(1, len(actions)):
            prev_a, curr_a = actions[i - 1], actions[i]
            if curr_a == prev_a:
                continue
            is_opposite_reversal = (
                (prev_a == ACTION_HEAT and curr_a == ACTION_COOL)
                or (prev_a == ACTION_COOL and curr_a == ACTION_HEAT)
            )
            is_chatter_cycle = i >= 2 and curr_a == actions[i - 2]
            if is_opposite_reversal or is_chatter_cycle:
                unnecessary_changes += 1

    # Stability: temperature settling step (earliest step staying Comfortable until end)
    classifications = df["classification"].tolist()
    step_nums = df["step"].tolist()
    settling_steps: Optional[int] = None
    if classifications and classifications[-1] == CLASS_COMFORTABLE:
        idx = len(classifications) - 1
        while idx > 0 and classifications[idx - 1] == CLASS_COMFORTABLE:
            idx -= 1
        settling_steps = int(step_nums[idx])
        settling_label = f"{settling_steps} steps"
    else:
        settling_label = f"Not settled ({n_steps} steps)"

    return AgentPerformanceMetrics(
        total_steps=n_steps,
        avg_abs_error=avg_abs_error,
        final_temp_error=final_temp_error,
        comfort_percentage=comfy_pct,
        comfort_steps=comfortable_steps if has_active else 0,
        overshoot_count=overshoots,
        total_energy=total_energy,
        avg_energy_per_step=avg_energy_per_step,
        heating_activations=heating_activations,
        cooling_activations=cooling_activations,
        off_steps=off_steps,
        temp_fluctuations=temp_fluctuations,
        unnecessary_action_changes=unnecessary_changes,
        settling_steps=settling_steps,
        settling_label=settling_label,
        cold_steps=cold_steps if has_active else 0,
        comfortable_steps=comfortable_steps if has_active else 0,
        hot_steps=hot_steps if has_active else 0,
        cold_percentage=cold_pct,
        comfortable_percentage=comfy_pct,
        hot_percentage=hot_pct,
    )


def determine_metric_winner(
    utility_value: float,
    reflex_value: float,
    higher_is_better: bool = False,
    tolerance: float = METRIC_TOLERANCE,
) -> str:
    """
    Determine the better-performing agent for a numeric metric without bias.
    """
    diff = float(utility_value) - float(reflex_value)
    if abs(diff) <= tolerance:
        return TIE_LABEL
    if higher_is_better:
        return AGENT_UTILITY if diff > 0 else AGENT_REFLEX
    return AGENT_UTILITY if diff < 0 else AGENT_REFLEX


def determine_settling_winner(
    utility_settling: Optional[int],
    reflex_settling: Optional[int],
) -> str:
    """Determine which agent settled faster into the comfort range."""
    if utility_settling is None and reflex_settling is None:
        return TIE_LABEL
    if utility_settling is not None and reflex_settling is None:
        return AGENT_UTILITY
    if utility_settling is None and reflex_settling is not None:
        return AGENT_REFLEX
    if utility_settling == reflex_settling:
        return TIE_LABEL
    return AGENT_UTILITY if utility_settling < reflex_settling else AGENT_REFLEX


def build_comparison_table_df(
    util_metrics: AgentPerformanceMetrics,
    refl_metrics: AgentPerformanceMetrics,
) -> pd.DataFrame:
    """
    Build the structured performance comparison DataFrame with columns:
    Performance Metric, Utility-Based Agent, Simple Reflex Agent, Better Performing Agent.
    """
    rows = [
        {
            "Performance Metric": "Average Absolute Temperature Error (°C)",
            "Utility-Based Agent": f"{util_metrics.avg_abs_error:.3f} °C",
            "Simple Reflex Agent": f"{refl_metrics.avg_abs_error:.3f} °C",
            "Better Performing Agent": determine_metric_winner(
                util_metrics.avg_abs_error, refl_metrics.avg_abs_error, higher_is_better=False
            ),
        },
        {
            "Performance Metric": "Final Temperature Error (°C)",
            "Utility-Based Agent": f"{util_metrics.final_temp_error:.3f} °C",
            "Simple Reflex Agent": f"{refl_metrics.final_temp_error:.3f} °C",
            "Better Performing Agent": determine_metric_winner(
                util_metrics.final_temp_error, refl_metrics.final_temp_error, higher_is_better=False
            ),
        },
        {
            "Performance Metric": "Time in Comfortable Range (%)",
            "Utility-Based Agent": f"{util_metrics.comfort_percentage:.1f}% ({util_metrics.comfort_steps} steps)",
            "Simple Reflex Agent": f"{refl_metrics.comfort_percentage:.1f}% ({refl_metrics.comfort_steps} steps)",
            "Better Performing Agent": determine_metric_winner(
                util_metrics.comfort_percentage, refl_metrics.comfort_percentage, higher_is_better=True
            ),
        },
        {
            "Performance Metric": "Number of Temperature Overshoots",
            "Utility-Based Agent": str(util_metrics.overshoot_count),
            "Simple Reflex Agent": str(refl_metrics.overshoot_count),
            "Better Performing Agent": determine_metric_winner(
                util_metrics.overshoot_count, refl_metrics.overshoot_count, higher_is_better=False
            ),
        },
        {
            "Performance Metric": "Total Energy Consumed (units)",
            "Utility-Based Agent": f"{util_metrics.total_energy:.1f} units",
            "Simple Reflex Agent": f"{refl_metrics.total_energy:.1f} units",
            "Better Performing Agent": determine_metric_winner(
                util_metrics.total_energy, refl_metrics.total_energy, higher_is_better=False
            ),
        },
        {
            "Performance Metric": "Average Energy per Step (units/step)",
            "Utility-Based Agent": f"{util_metrics.avg_energy_per_step:.3f} units/step",
            "Simple Reflex Agent": f"{refl_metrics.avg_energy_per_step:.3f} units/step",
            "Better Performing Agent": determine_metric_winner(
                util_metrics.avg_energy_per_step, refl_metrics.avg_energy_per_step, higher_is_better=False
            ),
        },
        {
            "Performance Metric": "Heating Activations (steps)",
            "Utility-Based Agent": str(util_metrics.heating_activations),
            "Simple Reflex Agent": str(refl_metrics.heating_activations),
            "Better Performing Agent": determine_metric_winner(
                util_metrics.heating_activations, refl_metrics.heating_activations, higher_is_better=False
            ),
        },
        {
            "Performance Metric": "Cooling Activations (steps)",
            "Utility-Based Agent": str(util_metrics.cooling_activations),
            "Simple Reflex Agent": str(refl_metrics.cooling_activations),
            "Better Performing Agent": determine_metric_winner(
                util_metrics.cooling_activations, refl_metrics.cooling_activations, higher_is_better=False
            ),
        },
        {
            "Performance Metric": "Temperature Fluctuations (°C/step)",
            "Utility-Based Agent": f"{util_metrics.temp_fluctuations:.3f} °C/step",
            "Simple Reflex Agent": f"{refl_metrics.temp_fluctuations:.3f} °C/step",
            "Better Performing Agent": determine_metric_winner(
                util_metrics.temp_fluctuations, refl_metrics.temp_fluctuations, higher_is_better=False
            ),
        },
        {
            "Performance Metric": "Unnecessary Action Changes (count)",
            "Utility-Based Agent": str(util_metrics.unnecessary_action_changes),
            "Simple Reflex Agent": str(refl_metrics.unnecessary_action_changes),
            "Better Performing Agent": determine_metric_winner(
                util_metrics.unnecessary_action_changes,
                refl_metrics.unnecessary_action_changes,
                higher_is_better=False,
            ),
        },
        {
            "Performance Metric": "Temperature Settling Behavior",
            "Utility-Based Agent": util_metrics.settling_label,
            "Simple Reflex Agent": refl_metrics.settling_label,
            "Better Performing Agent": determine_settling_winner(
                util_metrics.settling_steps, refl_metrics.settling_steps
            ),
        },
    ]
    return pd.DataFrame(rows)


def summarize_comparison(
    comparison_df: pd.DataFrame,
    util_metrics: AgentPerformanceMetrics,
    refl_metrics: AgentPerformanceMetrics,
) -> Dict[str, Any]:
    """
    Summarize the overall comparison purely from the recorded metric winners.
    """
    winners = comparison_df["Better Performing Agent"].tolist() if not comparison_df.empty else []
    utility_wins = winners.count(AGENT_UTILITY)
    reflex_wins = winners.count(AGENT_REFLEX)
    ties = winners.count(TIE_LABEL)

    if util_metrics.total_steps == 0:
        overall_winner = TIE_LABEL
        headline = "Run the synchronized simulation to compare both agents."
    elif utility_wins > reflex_wins:
        overall_winner = AGENT_UTILITY
        headline = (
            f"Utility-Based Agent led in {utility_wins} of {len(winners)} metrics "
            f"({reflex_wins} won by Simple Reflex Agent, {ties} tied)."
        )
    elif reflex_wins > utility_wins:
        overall_winner = AGENT_REFLEX
        headline = (
            f"Simple Reflex Agent led in {reflex_wins} of {len(winners)} metrics "
            f"({utility_wins} won by Utility-Based Agent, {ties} tied)."
        )
    else:
        overall_winner = TIE_LABEL
        headline = (
            f"Both agents tied overall ({utility_wins} metrics won each, {ties} tied)."
        )

    return {
        "overall_winner": overall_winner,
        "utility_wins": utility_wins,
        "reflex_wins": reflex_wins,
        "ties": ties,
        "total_metrics": len(winners),
        "headline": headline,
    }


class ComparisonRunner:
    """
    Manages a synchronized closed-loop comparison between UtilityBasedAgent
    and SimpleReflexAgent under identical conditions and disturbance sequences.
    """

    def __init__(self, config: ComparisonConfig) -> None:
        self.config = config
        self.reset(config)

    def reset(self, new_config: Optional[ComparisonConfig] = None) -> None:
        """Reset both environments, both agents, and the shared RNG disturbance stream."""
        if new_config is not None:
            self.config = new_config

        cfg = self.config
        self._rng = np.random.default_rng(cfg.random_seed)

        self.util_env = RoomEnvironment(
            initial_temp=cfg.initial_temp,
            desired_temp=cfg.desired_temp,
            outside_temp=cfg.outside_temp,
            insulation_level=cfg.insulation_level,
            heater_power=cfg.heater_power,
            cooling_power=cfg.cooling_power,
            disturbance_level=cfg.disturbance_level,
            comfort_tolerance=cfg.comfort_tolerance,
            random_seed=cfg.random_seed,
        )
        self.refl_env = RoomEnvironment(
            initial_temp=cfg.initial_temp,
            desired_temp=cfg.desired_temp,
            outside_temp=cfg.outside_temp,
            insulation_level=cfg.insulation_level,
            heater_power=cfg.heater_power,
            cooling_power=cfg.cooling_power,
            disturbance_level=cfg.disturbance_level,
            comfort_tolerance=cfg.comfort_tolerance,
            random_seed=cfg.random_seed,
        )

        self.util_agent = UtilityBasedAgent(
            weight_comfort=cfg.weight_comfort,
            weight_energy=cfg.weight_energy,
            weight_stability=cfg.weight_stability,
            weight_overshoot=cfg.weight_overshoot,
        )
        self.refl_agent = SimpleReflexAgent(
            lower_threshold=cfg.lower_threshold,
            upper_threshold=cfg.upper_threshold,
        )

        u_init = self.util_env.get_state()
        r_init = self.refl_env.get_state()
        lower_c = round(cfg.desired_temp - cfg.comfort_tolerance, 2)
        upper_c = round(cfg.desired_temp + cfg.comfort_tolerance, 2)
        init_err = round(abs(float(u_init["room_temp"]) - float(u_init["desired_temp"])), 3)

        self.utility_records: List[Dict[str, Any]] = [{
            "step": 0,
            "room_temp": float(u_init["room_temp"]),
            "outside_temp": float(u_init["outside_temp"]),
            "desired_temp": float(u_init["desired_temp"]),
            "lower_comfort": lower_c,
            "upper_comfort": upper_c,
            "action": ACTION_OFF,
            "predicted_temp": float(u_init["room_temp"]),
            "actual_temp": float(u_init["room_temp"]),
            "abs_error": init_err,
            "utility_score": 0.0,
            "energy_consumed": 0.0,
            "cumulative_energy": 0.0,
            "disturbance": 0.0,
            "classification": str(u_init["classification"]),
        }]

        self.reflex_records: List[Dict[str, Any]] = [{
            "step": 0,
            "room_temp": float(r_init["room_temp"]),
            "outside_temp": float(r_init["outside_temp"]),
            "desired_temp": float(r_init["desired_temp"]),
            "lower_comfort": lower_c,
            "upper_comfort": upper_c,
            "action": ACTION_OFF,
            "actual_temp": float(r_init["room_temp"]),
            "abs_error": init_err,
            "energy_consumed": 0.0,
            "cumulative_energy": 0.0,
            "disturbance": 0.0,
            "classification": str(r_init["classification"]),
        }]

    @property
    def current_step(self) -> int:
        """Return number of synchronized simulation steps completed."""
        return self.util_env.current_step

    @property
    def max_steps(self) -> int:
        """Return configured step horizon."""
        return int(self.config.max_steps)

    @property
    def is_finished(self) -> bool:
        """Return True when all configured steps have completed."""
        return self.current_step >= self.max_steps

    def step_once(self) -> bool:
        """
        Advance both agents by one synchronized step using an identical disturbance value.
        Returns True if a step executed, False if max_steps was already reached.
        """
        if self.is_finished:
            return False

        if self.config.disturbance_level > 0.0:
            disturbance = float(self._rng.normal(0.0, self.config.disturbance_level))
        else:
            disturbance = 0.0

        lower_c = round(self.config.desired_temp - self.config.comfort_tolerance, 2)
        upper_c = round(self.config.desired_temp + self.config.comfort_tolerance, 2)

        # 1. Utility-Based Agent step
        u_obs = self.util_env.get_state()
        u_dec = self.util_agent.decide_action(u_obs)
        u_next = self.util_env.step(u_dec.selected_action, disturbance=disturbance)
        self.utility_records.append({
            "step": int(u_next["step"]),
            "room_temp": float(u_obs["room_temp"]),
            "outside_temp": float(u_obs["outside_temp"]),
            "desired_temp": float(u_obs["desired_temp"]),
            "lower_comfort": lower_c,
            "upper_comfort": upper_c,
            "action": u_dec.selected_action,
            "predicted_temp": float(u_dec.predicted_temp),
            "actual_temp": float(u_next["room_temp"]),
            "abs_error": round(abs(float(u_next["room_temp"]) - float(u_obs["desired_temp"])), 3),
            "utility_score": float(u_dec.utility_score),
            "energy_consumed": float(u_dec.energy_consumed),
            "cumulative_energy": round(self.util_agent.total_energy_consumed, 2),
            "disturbance": round(disturbance, 4),
            "classification": str(u_next["classification"]),
        })

        # 2. Simple Reflex Agent step (identical disturbance!)
        r_rec = self.refl_agent.step_environment(self.refl_env, disturbance=disturbance)
        self.reflex_records.append({
            "step": int(r_rec["step"]),
            "room_temp": float(r_rec["room_temp"]),
            "outside_temp": float(r_rec["outside_temp"]),
            "desired_temp": float(r_rec["desired_temp"]),
            "lower_comfort": lower_c,
            "upper_comfort": upper_c,
            "action": r_rec["action"],
            "actual_temp": float(r_rec["actual_temp"]),
            "abs_error": round(abs(float(r_rec["actual_temp"]) - float(r_rec["desired_temp"])), 3),
            "energy_consumed": float(r_rec["energy_consumed"]),
            "cumulative_energy": float(r_rec["cumulative_energy"]),
            "disturbance": round(disturbance, 4),
            "classification": str(r_rec["classification"]),
        })

        return True

    def run_remaining(self) -> int:
        """Execute all remaining steps until max_steps is reached."""
        steps_executed = 0
        while self.step_once():
            steps_executed += 1
        return steps_executed

    def run_all(self) -> int:
        """Reset and run the full synchronized comparison simulation."""
        self.reset()
        return self.run_remaining()

    def get_utility_df(self) -> pd.DataFrame:
        """Return Utility-Based Agent recorded comparison telemetry."""
        return pd.DataFrame(self.utility_records)

    def get_reflex_df(self) -> pd.DataFrame:
        """Return Simple Reflex Agent recorded comparison telemetry."""
        return pd.DataFrame(self.reflex_records)

    def get_combined_df(self) -> pd.DataFrame:
        """Return side-by-side step telemetry for both agents in one DataFrame."""
        u_df = self.get_utility_df()
        r_df = self.get_reflex_df()
        if u_df.empty or r_df.empty:
            return pd.DataFrame()

        return pd.DataFrame({
            "step": u_df["step"],
            "outside_temp": u_df["outside_temp"],
            "desired_temp": u_df["desired_temp"],
            "lower_comfort": u_df["lower_comfort"],
            "upper_comfort": u_df["upper_comfort"],
            "disturbance": u_df["disturbance"],
            "utility_room_temp": u_df["room_temp"],
            "utility_action": u_df["action"],
            "utility_predicted_temp": u_df["predicted_temp"],
            "utility_actual_temp": u_df["actual_temp"],
            "utility_abs_error": u_df["abs_error"],
            "utility_score": u_df["utility_score"],
            "utility_energy_consumed": u_df["energy_consumed"],
            "utility_cumulative_energy": u_df["cumulative_energy"],
            "utility_classification": u_df["classification"],
            "reflex_room_temp": r_df["room_temp"],
            "reflex_action": r_df["action"],
            "reflex_actual_temp": r_df["actual_temp"],
            "reflex_abs_error": r_df["abs_error"],
            "reflex_energy_consumed": r_df["energy_consumed"],
            "reflex_cumulative_energy": r_df["cumulative_energy"],
            "reflex_classification": r_df["classification"],
        })

    def evaluate(self) -> Dict[str, Any]:
        """Compute metrics, comparison DataFrame, and winner summary from recorded data."""
        u_df = self.get_utility_df()
        r_df = self.get_reflex_df()
        util_metrics = compute_agent_metrics(u_df)
        refl_metrics = compute_agent_metrics(r_df)
        comparison_df = build_comparison_table_df(util_metrics, refl_metrics)
        summary = summarize_comparison(comparison_df, util_metrics, refl_metrics)
        return {
            "utility_df": u_df,
            "reflex_df": r_df,
            "combined_df": self.get_combined_df(),
            "utility_metrics": util_metrics,
            "reflex_metrics": refl_metrics,
            "comparison_df": comparison_df,
            "summary": summary,
        }

