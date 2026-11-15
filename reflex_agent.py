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
    HVACControlDefaults,
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
    "people_present",
    "num_people",
    "room_size",
    "delta_occupancy",
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
        upper_heating_threshold: Optional[float] = None,
        lower_cooling_threshold: Optional[float] = None,
        min_on_duration: Optional[int] = None,
        min_off_duration: Optional[int] = None,
        safety_min_temp: float = HVACControlDefaults.CRITICAL_SAFETY_MIN_TEMP,
        safety_max_temp: float = HVACControlDefaults.CRITICAL_SAFETY_MAX_TEMP,
    ) -> None:
        """
        Initialize the simple reflex thermostat with lower and upper thresholds,
        hysteresis cut-off thresholds, minimum operating durations, and safety limits.

        Args:
            lower_threshold: Room temperature below which HEAT is activated (°C).
            upper_threshold: Room temperature above which COOL is activated (°C).
            energy_cost_heat: Energy consumption unit when heating.
            energy_cost_cool: Energy consumption unit when cooling.
            energy_cost_off: Energy consumption unit when HVAC is off.
            upper_heating_threshold: Upper temperature threshold at which active heating shuts OFF (°C).
            lower_cooling_threshold: Lower temperature threshold at which active cooling shuts OFF (°C).
            min_on_duration: Minimum consecutive timesteps heating/cooling must run before deactivation.
            min_off_duration: Minimum consecutive timesteps HVAC must remain OFF before re-activation.
            safety_min_temp: Critical freeze protection safety floor (°C, overrides dwell time).
            safety_max_temp: Critical overheat protection safety ceiling (°C, overrides dwell time).

        Raises:
            ValueError: If lower_threshold >= upper_threshold.
        """
        lower, upper = self._validate_thresholds(lower_threshold, upper_threshold)
        self.lower_threshold: float = lower
        self.upper_threshold: float = upper
        self.upper_heating_threshold: Optional[float] = (
            float(upper_heating_threshold) if upper_heating_threshold is not None else None
        )
        self.lower_cooling_threshold: Optional[float] = (
            float(lower_cooling_threshold) if lower_cooling_threshold is not None else None
        )

        has_hysteresis = (upper_heating_threshold is not None or lower_cooling_threshold is not None)
        if min_on_duration is not None:
            self.min_on_duration = int(min_on_duration)
        else:
            self.min_on_duration = HVACControlDefaults.MIN_ON_DURATION_STEPS if has_hysteresis else 1

        if min_off_duration is not None:
            self.min_off_duration = int(min_off_duration)
        else:
            self.min_off_duration = HVACControlDefaults.MIN_OFF_DURATION_STEPS if has_hysteresis else 1

        self.safety_min_temp = float(safety_min_temp)
        self.safety_max_temp = float(safety_max_temp)
        self.steps_in_current_action: int = 0

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
        self.steps_in_current_action = 0

    def update_thresholds(
        self,
        lower_threshold: float,
        upper_threshold: float,
        upper_heating_threshold: Optional[float] = None,
        lower_cooling_threshold: Optional[float] = None,
    ) -> None:
        """
        Update lower and upper temperature thresholds and optional hysteresis cutoffs.

        Args:
            lower_threshold: New lower threshold (°C).
            upper_threshold: New upper threshold (°C).
            upper_heating_threshold: Optional new upper heating shut-off threshold (°C).
            lower_cooling_threshold: Optional new lower cooling shut-off threshold (°C).

        Raises:
            ValueError: If lower_threshold >= upper_threshold.
        """
        lower, upper = self._validate_thresholds(lower_threshold, upper_threshold)
        self.lower_threshold = lower
        self.upper_threshold = upper
        if upper_heating_threshold is not None:
            self.upper_heating_threshold = float(upper_heating_threshold)
        if lower_cooling_threshold is not None:
            self.lower_cooling_threshold = float(lower_cooling_threshold)

    def get_rules(self) -> List[Dict[str, str]]:
        """Return condition-action rules formatted with the active thresholds and hysteresis."""
        heat_cutoff = self.upper_heating_threshold if self.upper_heating_threshold is not None else self.lower_threshold
        cool_cutoff = self.lower_cooling_threshold if self.lower_cooling_threshold is not None else self.upper_threshold

        heat_lbl = f"< {self.lower_threshold:.1f}°C → {ACTION_HEAT}"
        if self.upper_heating_threshold is not None:
            heat_lbl += f" (until {heat_cutoff:.1f}°C reached)"

        cool_lbl = f"> {self.upper_threshold:.1f}°C → {ACTION_COOL}"
        if self.lower_cooling_threshold is not None:
            cool_lbl += f" (until {cool_cutoff:.1f}°C reached)"

        return [
            {
                "condition": f"< {self.lower_threshold:.1f}°C",
                "action": ACTION_HEAT,
                "label": heat_lbl,
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
                "label": cool_lbl,
                "description": "Turn on cooling",
            },
        ]

    def select_action(
        self,
        room_temp: float,
        prev_action: Optional[str] = None,
        steps_in_action: Optional[int] = None,
    ) -> str:
        """
        Apply condition-action rules using the current observed room temperature and prior state.

        Hysteresis & Dwell Control:
            1. Critical Safety Limits:
               IF room_temp <= safety_min_temp (12°C): HEAT immediately (freeze protection)
               IF room_temp >= safety_max_temp (35°C): COOL immediately (overheat protection)
            2. Minimum Operating Duration:
               Hold current state if steps_in_action < min_duration (unless safety breached)
            3. Hysteresis Deactivation:
               IF prior is HEAT: continue heating until upper_heating_threshold reached
               IF prior is COOL: continue cooling until lower_cooling_threshold reached
            4. Deadband:
               IF prior is OFF: remain OFF within [lower_threshold, upper_threshold]
        """
        temp = float(room_temp)
        prior = prev_action if prev_action is not None else self.last_action
        steps = steps_in_action if steps_in_action is not None else self.steps_in_current_action

        # 1. Critical safety override (independent of minimum duration or deadband)
        if temp <= self.safety_min_temp:
            return ACTION_HEAT
        if temp >= self.safety_max_temp:
            return ACTION_COOL

        heat_cutoff = self.upper_heating_threshold if self.upper_heating_threshold is not None else self.lower_threshold
        cool_cutoff = self.lower_cooling_threshold if self.lower_cooling_threshold is not None else self.upper_threshold

        if prior == ACTION_HEAT:
            # Minimum run duration requirement
            if steps < self.min_on_duration:
                if temp < self.upper_threshold:
                    return ACTION_HEAT
                else:
                    return ACTION_OFF
            # Minimum run duration satisfied: evaluate hysteresis deactivation
            if temp < heat_cutoff:
                return ACTION_HEAT
            else:
                return ACTION_OFF

        elif prior == ACTION_COOL:
            # Minimum run duration requirement
            if steps < self.min_on_duration:
                if temp > self.lower_threshold:
                    return ACTION_COOL
                else:
                    return ACTION_OFF
            # Minimum run duration satisfied: evaluate hysteresis deactivation
            if temp > cool_cutoff:
                return ACTION_COOL
            else:
                return ACTION_OFF

        else:  # prior == ACTION_OFF or None
            # Minimum OFF duration requirement (prevent rapid restart)
            if prior == ACTION_OFF and steps < self.min_off_duration:
                if self.lower_threshold <= temp <= self.upper_threshold:
                    return ACTION_OFF

            if temp < self.lower_threshold:
                return ACTION_HEAT
            elif temp > self.upper_threshold:
                return ACTION_COOL
            else:
                return ACTION_OFF

    def _build_decision(
        self,
        current_temp: float,
        step: int,
        prev_action: Optional[str] = None,
        steps_in_action: Optional[int] = None,
    ) -> ReflexDecision:
        """Evaluate current room temperature against thresholds and construct ReflexDecision."""
        prior = prev_action
        steps = steps_in_action if steps_in_action is not None else self.steps_in_current_action
        if prior is None and self.last_decision is not None:
            prev_step = self.last_decision.step
            prev_temp = self.last_decision.room_temp
            if step > prev_step and abs(current_temp - prev_temp) <= 3.5:
                prior = self.last_decision.selected_action
                if steps_in_action is None:
                    steps = self.steps_in_current_action
            else:
                steps = 0
        elif prior is not None and prior != self.last_action:
            steps = 0

        selected_action = self.select_action(current_temp, prev_action=prior, steps_in_action=steps)
        step_energy = self.energy_costs[selected_action]

        heat_cutoff = self.upper_heating_threshold if self.upper_heating_threshold is not None else self.lower_threshold
        cool_cutoff = self.lower_cooling_threshold if self.lower_cooling_threshold is not None else self.upper_threshold

        # Determine triggering rule & explanatory reason
        if current_temp <= self.safety_min_temp:
            rule_triggered = f"≤ {self.safety_min_temp:.1f}°C (Safety Limit) → {ACTION_HEAT}"
            reason = (
                f"Critical safety limit breached: Room temperature ({current_temp:.2f}°C) is below "
                f"freeze protection limit ({self.safety_min_temp:.1f}°C). Emergency HEAT activated."
            )
        elif current_temp >= self.safety_max_temp:
            rule_triggered = f"≥ {self.safety_max_temp:.1f}°C (Safety Limit) → {ACTION_COOL}"
            reason = (
                f"Critical safety limit breached: Room temperature ({current_temp:.2f}°C) is above "
                f"overheat protection limit ({self.safety_max_temp:.1f}°C). Emergency COOL activated."
            )
        elif selected_action == ACTION_HEAT:
            if prior == ACTION_HEAT and steps < self.min_on_duration:
                rule_triggered = f"Min Run Duration ({steps}/{self.min_on_duration}) → {ACTION_HEAT}"
                reason = (
                    f"Room temperature ({current_temp:.2f}°C): maintaining HEAT to satisfy minimum "
                    f"operating duration ({steps}/{self.min_on_duration} steps), preventing short-cycling."
                )
            elif prior == ACTION_HEAT and self.upper_heating_threshold is not None:
                rule_triggered = f"< {heat_cutoff:.1f}°C (Heating to cutoff) → {ACTION_HEAT}"
                reason = (
                    f"Room temperature ({current_temp:.2f}°C) is heating toward upper deactivation "
                    f"threshold ({heat_cutoff:.1f}°C), maintaining HEAT."
                )
            else:
                rule_triggered = f"< {self.lower_threshold:.1f}°C → {ACTION_HEAT}"
                reason = (
                    f"Current room temperature ({current_temp:.2f}°C) is below lower activation "
                    f"boundary ({self.lower_threshold:.1f}°C), triggering HEAT."
                )
        elif selected_action == ACTION_COOL:
            if prior == ACTION_COOL and steps < self.min_on_duration:
                rule_triggered = f"Min Run Duration ({steps}/{self.min_on_duration}) → {ACTION_COOL}"
                reason = (
                    f"Room temperature ({current_temp:.2f}°C): maintaining COOL to satisfy minimum "
                    f"operating duration ({steps}/{self.min_on_duration} steps), preventing short-cycling."
                )
            elif prior == ACTION_COOL and self.lower_cooling_threshold is not None:
                rule_triggered = f"> {cool_cutoff:.1f}°C (Cooling to cutoff) → {ACTION_COOL}"
                reason = (
                    f"Room temperature ({current_temp:.2f}°C) is cooling toward lower deactivation "
                    f"threshold ({cool_cutoff:.1f}°C), maintaining COOL."
                )
            else:
                rule_triggered = f"> {self.upper_threshold:.1f}°C → {ACTION_COOL}"
                reason = (
                    f"Current room temperature ({current_temp:.2f}°C) is above upper activation "
                    f"boundary ({self.upper_threshold:.1f}°C), triggering COOL."
                )
        else:  # selected_action == ACTION_OFF
            if prior == ACTION_OFF and steps < self.min_off_duration:
                rule_triggered = f"Min Dwell Duration ({steps}/{self.min_off_duration}) → {ACTION_OFF}"
                reason = (
                    f"Room temperature ({current_temp:.2f}°C): maintaining OFF to satisfy minimum "
                    f"compressor dwell duration ({steps}/{self.min_off_duration} steps), avoiding hunting."
                )
            elif prior == ACTION_HEAT and self.upper_heating_threshold is not None:
                rule_triggered = f"≥ {heat_cutoff:.1f}°C (Heating cutoff reached) → {ACTION_OFF}"
                reason = (
                    f"Room temperature ({current_temp:.2f}°C) reached upper heating threshold "
                    f"({heat_cutoff:.1f}°C), turning heating OFF."
                )
            elif prior == ACTION_COOL and self.lower_cooling_threshold is not None:
                rule_triggered = f"≤ {cool_cutoff:.1f}°C (Cooling cutoff reached) → {ACTION_OFF}"
                reason = (
                    f"Room temperature ({current_temp:.2f}°C) reached lower cooling threshold "
                    f"({cool_cutoff:.1f}°C), turning cooling OFF."
                )
            else:
                rule_triggered = (
                    f"{self.lower_threshold:.1f}°C–{self.upper_threshold:.1f}°C (Deadband) → {ACTION_OFF}"
                )
                reason = (
                    f"Current room temperature ({current_temp:.2f}°C) is within comfortable deadband "
                    f"[{self.lower_threshold:.1f}°C, {self.upper_threshold:.1f}°C], keeping HVAC OFF."
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
        prior = observation.get("prev_action", self.last_action)
        steps = observation.get("steps_in_action", self.steps_in_current_action)
        return self._build_decision(current_temp=current_temp, step=step, prev_action=prior, steps_in_action=steps)

    def decide_action(self, observation: Dict[str, Any]) -> ReflexDecision:
        """
        Execute the percept-to-action reflex rule for one time step:
        1. Observe current room temperature.
        2. Compare against lower and upper thresholds with hysteresis and dwell times.
        3. Select HEAT, COOL, or OFF via condition-action rule.
        4. Update agent action and energy tracking.
        """
        current_temp = float(observation["room_temp"])
        step = int(observation.get("step", len(self.action_history)))
        prior = observation.get("prev_action", self.last_action)
        steps = observation.get("steps_in_action", self.steps_in_current_action)

        decision = self._build_decision(current_temp=current_temp, step=step, prev_action=prior, steps_in_action=steps)

        if self.last_decision is not None and self.last_decision.selected_action == decision.selected_action:
            self.steps_in_current_action += 1
        else:
            self.steps_in_current_action = 1

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
            "people_present": bool(next_state.get("people_present", True)),
            "num_people": int(next_state.get("num_people", 2)),
            "room_size": float(next_state.get("room_size", 25.0)),
            "delta_occupancy": float(next_state.get("delta_occupancy", 0.0)),
        }
        self.simulation_history.append(record)
        return record

    @property
    def last_action(self) -> Optional[str]:
        """Return the most recently selected action, or None if no decision has been made."""
        if self.last_decision is not None:
            return self.last_decision.selected_action
        if self.action_history:
            return self.action_history[-1]
        return None

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

