"""
Simple Reflex Thermostat Agent for Room Temperature Control.

Implements a pure condition-action (simple reflex) agent that selects HVAC actions
strictly from the current observed room temperature compared against configurable
lower and upper temperature thresholds.

Rules:
    IF room temperature < lower_threshold:
        HEAT
    ELSE IF room temperature > upper_threshold:
        COOL
    ELSE:
        OFF

This agent does NOT calculate utility scores, predict future temperatures,
or use learning mechanisms.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from config import (
    ACTION_COOL,
    ACTION_HEAT,
    ACTION_OFF,
    ReflexThermostatDefaults,
)
from environment import RoomEnvironment


HISTORY_COLUMNS = [
    "step",
    "room_temp",
    "outside_temp",
    "desired_temp",
    "action",
    "actual_temp",
    "energy_consumed",
    "cumulative_energy",
    "classification",
]


@dataclass
class ReflexDecision:
    """Summary of a single Simple Reflex Thermostat decision."""

    step: int
    room_temp: float
    lower_threshold: float
    upper_threshold: float
    selected_action: str
    energy_consumed: float
    rule_triggered: str
    reason: str


class SimpleReflexAgent:
    """
    Simple reflex thermostat that controls room temperature using predefined
    condition-action rules over the current room temperature percept only.
    """

    def __init__(
        self,
        lower_threshold: float = ReflexThermostatDefaults.DEFAULT_LOWER_THRESHOLD,
        upper_threshold: float = ReflexThermostatDefaults.DEFAULT_UPPER_THRESHOLD,
        energy_cost_heat: float = ReflexThermostatDefaults.ENERGY_COST_HEAT,
        energy_cost_cool: float = ReflexThermostatDefaults.ENERGY_COST_COOL,
        energy_cost_off: float = ReflexThermostatDefaults.ENERGY_COST_OFF,
    ) -> None:
        """
        Initialize the simple reflex thermostat with lower and upper thresholds.

        Args:
            lower_threshold: Room temperature below which HEAT is activated (°C).
            upper_threshold: Room temperature above which COOL is activated (°C).
            energy_cost_heat: Energy consumption unit when heating.
            energy_cost_cool: Energy consumption unit when cooling.
            energy_cost_off: Energy consumption unit when HVAC is off.

        Raises:
            ValueError: If lower_threshold >= upper_threshold.
        """
        lower, upper = self._validate_thresholds(lower_threshold, upper_threshold)
        self.lower_threshold: float = lower
        self.upper_threshold: float = upper

        self.energy_costs: Dict[str, float] = {
            ACTION_HEAT: float(energy_cost_heat),
            ACTION_COOL: float(energy_cost_cool),
            ACTION_OFF: float(energy_cost_off),
        }

        # Independent state & simulation tracking
        self.last_decision: Optional[ReflexDecision] = None
        self.total_energy_consumed: float = 0.0
        self.action_history: List[str] = []
        self.simulation_history: List[Dict[str, Any]] = []

    @staticmethod
    def _validate_thresholds(
        lower_threshold: float,
        upper_threshold: float,
    ) -> Tuple[float, float]:
        """Ensure lower threshold is strictly less than upper threshold."""
        lower = float(lower_threshold)
        upper = float(upper_threshold)
        if lower >= upper:
            raise ValueError(
                f"Lower threshold ({lower:.1f}°C) must be strictly less than "
                f"upper threshold ({upper:.1f}°C)."
            )
        return lower, upper

    def reset(self) -> None:
        """Reset cumulative energy and simulation history."""
        self.last_decision = None
        self.total_energy_consumed = 0.0
        self.action_history = []
        self.simulation_history = []

    def update_thresholds(self, lower_threshold: float, upper_threshold: float) -> None:
        """
        Update lower and upper temperature thresholds.

        Args:
            lower_threshold: New lower threshold (°C).
            upper_threshold: New upper threshold (°C).

        Raises:
            ValueError: If lower_threshold >= upper_threshold.
        """
        lower, upper = self._validate_thresholds(lower_threshold, upper_threshold)
        self.lower_threshold = lower
        self.upper_threshold = upper

    def get_rules(self) -> List[Dict[str, str]]:
        """Return condition-action rules formatted with the active thresholds."""
        return [
            {
                "condition": f"< {self.lower_threshold:.1f}°C",
                "action": ACTION_HEAT,
                "label": f"< {self.lower_threshold:.1f}°C → {ACTION_HEAT}",
                "description": "Turn on heating",
            },
            {
                "condition": f"{self.lower_threshold:.1f}°C–{self.upper_threshold:.1f}°C",
                "action": ACTION_OFF,
                "label": f"{self.lower_threshold:.1f}°C–{self.upper_threshold:.1f}°C → {ACTION_OFF}",
                "description": "Turn off both systems",
            },
            {
                "condition": f"> {self.upper_threshold:.1f}°C",
                "action": ACTION_COOL,
                "label": f"> {self.upper_threshold:.1f}°C → {ACTION_COOL}",
                "description": "Turn on cooling",
            },
        ]

    def select_action(self, room_temp: float) -> str:
        """
        Apply the condition-action rule using only the current observed room temperature.

        IF room_temp < lower_threshold -> HEAT
        ELSE IF room_temp > upper_threshold -> COOL
        ELSE -> OFF
        """
        temp = float(room_temp)
        if temp < self.lower_threshold:
            return ACTION_HEAT
        elif temp > self.upper_threshold:
            return ACTION_COOL
        else:
            return ACTION_OFF

    def _build_decision(self, current_temp: float, step: int) -> ReflexDecision:
        """Evaluate current room temperature against thresholds and construct ReflexDecision."""
        selected_action = self.select_action(current_temp)
        step_energy = self.energy_costs[selected_action]

        if selected_action == ACTION_HEAT:
            rule_triggered = f"< {self.lower_threshold:.1f}°C → {ACTION_HEAT}"
            reason = (
                f"Current room temperature ({current_temp:.2f}°C) is below the lower "
                f"threshold ({self.lower_threshold:.1f}°C), triggering HEAT."
            )
        elif selected_action == ACTION_COOL:
            rule_triggered = f"> {self.upper_threshold:.1f}°C → {ACTION_COOL}"
            reason = (
                f"Current room temperature ({current_temp:.2f}°C) is above the upper "
                f"threshold ({self.upper_threshold:.1f}°C), triggering COOL."
            )
        else:
            rule_triggered = (
                f"{self.lower_threshold:.1f}°C–{self.upper_threshold:.1f}°C → {ACTION_OFF}"
            )
            reason = (
                f"Current room temperature ({current_temp:.2f}°C) is within "
                f"[{self.lower_threshold:.1f}°C, {self.upper_threshold:.1f}°C], "
                f"keeping both heating and cooling OFF."
            )

        return ReflexDecision(
            step=step,
            room_temp=round(current_temp, 2),
            lower_threshold=self.lower_threshold,
            upper_threshold=self.upper_threshold,
            selected_action=selected_action,
            energy_consumed=step_energy,
            rule_triggered=rule_triggered,
            reason=reason,
        )

    def preview_decision(self, observation: Dict[str, Any]) -> ReflexDecision:
        """
        Inspect which condition-action rule applies to the current observation
        without mutating energy counters or simulation history.
        """
        current_temp = float(observation["room_temp"])
        step = int(observation.get("step", len(self.action_history)))
        return self._build_decision(current_temp=current_temp, step=step)

    def decide_action(self, observation: Dict[str, Any]) -> ReflexDecision:
        """
        Execute the percept-to-action reflex rule for one time step:
        1. Observe current room temperature.
        2. Compare against lower and upper thresholds.
        3. Select HEAT, COOL, or OFF via condition-action rule.
        4. Update agent action and energy tracking.
        """
        current_temp = float(observation["room_temp"])
        step = int(observation.get("step", len(self.action_history)))

        decision = self._build_decision(current_temp=current_temp, step=step)

        self.total_energy_consumed += decision.energy_consumed
        self.action_history.append(decision.selected_action)
        self.last_decision = decision

        return decision

    def record_step(
        self,
        observation: Dict[str, Any],
        decision: ReflexDecision,
        next_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Record step telemetry after sending the selected action to the environment.
        """
        record = {
            "step": int(next_state["step"]),
            "room_temp": float(observation["room_temp"]),
            "outside_temp": float(observation["outside_temp"]),
            "desired_temp": float(observation["desired_temp"]),
            "action": decision.selected_action,
            "actual_temp": float(next_state["room_temp"]),
            "energy_consumed": float(decision.energy_consumed),
            "cumulative_energy": round(self.total_energy_consumed, 2),
            "classification": str(next_state["classification"]),
        }
        self.simulation_history.append(record)
        return record

    def step_environment(
        self,
        env: RoomEnvironment,
        disturbance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute one full simulation step against a RoomEnvironment instance:
        1. Observe the current room temperature.
        2. Compare it against the predefined thresholds.
        3. Apply the corresponding condition-action rule.
        4. Send the selected action to the environment.
        5. Record the action and resulting temperature.
        """
        observation = env.get_state()
        decision = self.decide_action(observation)
        next_state = env.step(decision.selected_action, disturbance=disturbance)
        return self.record_step(observation, decision, next_state)

    def get_history_df(self) -> pd.DataFrame:
        """Return simulation step history as a pandas DataFrame."""
        if not self.simulation_history:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        return pd.DataFrame(self.simulation_history)

    def get_metrics(self) -> Dict[str, Any]:
        """Return cumulative reflex agent action and energy statistics."""
        total_steps = len(self.action_history)
        heat_steps = self.action_history.count(ACTION_HEAT)
        cool_steps = self.action_history.count(ACTION_COOL)
        off_steps = self.action_history.count(ACTION_OFF)

        return {
            "total_steps": total_steps,
            "total_energy": round(self.total_energy_consumed, 1),
            "heat_count": heat_steps,
            "cool_count": cool_steps,
            "off_count": off_steps,
            "heat_percentage": round((heat_steps / total_steps * 100) if total_steps else 0.0, 1),
            "cool_percentage": round((cool_steps / total_steps * 100) if total_steps else 0.0, 1),
            "off_percentage": round((off_steps / total_steps * 100) if total_steps else 0.0, 1),
            "last_action": self.last_decision.selected_action if self.last_decision else ACTION_OFF,
        }

