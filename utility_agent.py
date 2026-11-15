"""
Utility-Based Intelligent Agent for Room Temperature Control.

Implements an intelligent agent that:
1. Predicts future thermal outcomes using Newton's law of cooling.
2. Evaluates candidate actions (HEAT, COOL, OFF) across multiple competing objectives:
   - Temperature Comfort Penalty
   - Energy Consumption Cost
   - Temperature Instability Penalty
   - Comfort Zone Overshoot Penalty
3. Selects the action maximizing the net weighted utility function:
   Utility(a) = - (w_comfort * P_comfort) - (w_energy * C_energy)
                - (w_stability * P_stability) - (w_overshoot * P_overshoot)
4. Generates interpretable natural language explanations for every decision.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

from config import (
    ACTION_COOL,
    ACTION_HEAT,
    ACTION_OFF,
    ALL_ACTIONS,
    HVACControlDefaults,
    INSULATION_FACTORS,
    UtilityWeightsDefaults,
)


@dataclass
class ActionEvaluation:
    """Detailed evaluation metrics for a single candidate action."""
    action: str
    predicted_temp: float
    comfort_penalty: float
    energy_cost: float
    instability_penalty: float
    overshoot_penalty: float
    weighted_comfort: float
    weighted_energy: float
    weighted_stability: float
    weighted_overshoot: float
    utility: float
    delta_occupancy: float = 0.0
    room_scale: float = 1.0
    occupancy_comfort_factor: float = 1.0
    is_occupied: bool = True


@dataclass
class AgentDecision:
    """Complete summary of the agent's decision at a given time step."""
    step: int
    current_temp: float
    desired_temp: float
    outside_temp: float
    selected_action: str
    predicted_temp: float
    utility_score: float
    energy_consumed: float
    reason: str
    action_evaluations: Dict[str, ActionEvaluation] = field(default_factory=dict)
    people_present: bool = True
    num_people: int = 2
    room_size: float = 25.0
    delta_occupancy: float = 0.0


class UtilityBasedAgent:
    """
    Intelligent agent that controls room temperature by maximizing a multi-factor
    utility function over projected future states.
    """

    def __init__(
        self,
        weight_comfort: float = UtilityWeightsDefaults.DEFAULT_WEIGHT_COMFORT,
        weight_energy: float = UtilityWeightsDefaults.DEFAULT_WEIGHT_ENERGY,
        weight_stability: float = UtilityWeightsDefaults.DEFAULT_WEIGHT_STABILITY,
        weight_overshoot: float = UtilityWeightsDefaults.DEFAULT_WEIGHT_OVERSHOOT,
        energy_cost_heat: float = UtilityWeightsDefaults.ENERGY_COST_HEAT,
        energy_cost_cool: float = UtilityWeightsDefaults.ENERGY_COST_COOL,
        energy_cost_off: float = UtilityWeightsDefaults.ENERGY_COST_OFF,
        unoccupied_min_temp: float = HVACControlDefaults.DEFAULT_UNOCCUPIED_MIN_TEMP,
        unoccupied_max_temp: float = HVACControlDefaults.DEFAULT_UNOCCUPIED_MAX_TEMP,
        safety_min_temp: float = HVACControlDefaults.CRITICAL_SAFETY_MIN_TEMP,
        safety_max_temp: float = HVACControlDefaults.CRITICAL_SAFETY_MAX_TEMP,
        min_on_duration: int = HVACControlDefaults.MIN_ON_DURATION_STEPS,
        min_off_duration: int = HVACControlDefaults.MIN_OFF_DURATION_STEPS,
    ) -> None:
        """
        Initialize the utility-based agent with configurable weights, costs,
        occupancy setback boundaries, dwell durations, and safety limits.

        Args:
            weight_comfort: Importance weight for proximity to target temperature.
            weight_energy: Importance weight for minimizing power consumption.
            weight_stability: Importance weight for minimizing abrupt temperature shifts.
            weight_overshoot: Importance weight for avoiding comfort zone violations.
            energy_cost_heat: Energy cost unit for operating the heater.
            energy_cost_cool: Energy cost unit for operating the cooler.
            energy_cost_off: Energy cost unit when HVAC is disabled.
            unoccupied_min_temp: Lower boundary for unoccupied setback mode (°C).
            unoccupied_max_temp: Upper boundary for unoccupied setback mode (°C).
            safety_min_temp: Critical freeze protection safety floor (°C).
            safety_max_temp: Critical overheat protection safety ceiling (°C).
            min_on_duration: Minimum consecutive timesteps heating/cooling must run.
            min_off_duration: Minimum consecutive timesteps HVAC must stay OFF.
        """
        self.weight_comfort = float(weight_comfort)
        self.weight_energy = float(weight_energy)
        self.weight_stability = float(weight_stability)
        self.weight_overshoot = float(weight_overshoot)

        self.unoccupied_min_temp = float(unoccupied_min_temp)
        self.unoccupied_max_temp = float(unoccupied_max_temp)
        self.safety_min_temp = float(safety_min_temp)
        self.safety_max_temp = float(safety_max_temp)
        self.min_on_duration = int(min_on_duration)
        self.min_off_duration = int(min_off_duration)

        self.energy_costs = {
            ACTION_HEAT: float(energy_cost_heat),
            ACTION_COOL: float(energy_cost_cool),
            ACTION_OFF: float(energy_cost_off),
        }

        # Tracking state
        self.last_decision: Optional[AgentDecision] = None
        self.total_energy_consumed: float = 0.0
        self.action_history: List[str] = []
        self.decision_history: List[Dict[str, Any]] = []
        self.steps_in_current_action: int = 0

    def reset(self) -> None:
        """Reset agent memory and cumulative metrics."""
        self.last_decision = None
        self.total_energy_consumed = 0.0
        self.action_history = []
        self.decision_history = []
        self.steps_in_current_action = 0

    @property
    def last_action(self) -> Optional[str]:
        """Return the most recently selected action, or None if no decision has been made."""
        if self.last_decision is not None:
            return self.last_decision.selected_action
        if self.action_history:
            return self.action_history[-1]
        return None

    def update_weights(
        self,
        weight_comfort: Optional[float] = None,
        weight_energy: Optional[float] = None,
        weight_stability: Optional[float] = None,
        weight_overshoot: Optional[float] = None,
    ) -> None:
        """Update utility weights dynamically from user interface configuration."""
        if weight_comfort is not None:
            self.weight_comfort = float(weight_comfort)
        if weight_energy is not None:
            self.weight_energy = float(weight_energy)
        if weight_stability is not None:
            self.weight_stability = float(weight_stability)
        if weight_overshoot is not None:
            self.weight_overshoot = float(weight_overshoot)

    def predict_temperature(
        self,
        current_temp: float,
        outside_temp: float,
        insulation_level: str,
        action: str,
        heater_power: float,
        cooling_power: float,
        people_present: bool = True,
        num_people: int = 2,
        room_size: float = 25.0,
        heat_gain_per_person: float = 0.08,
    ) -> float:
        """
        Predict the next room temperature for a candidate action using the
        physical thermodynamic model of the environment, incorporating room size
        and internal heat gain from occupants.

        Model:
            delta_ambient = k * (outside_temp - current_temp)
            room_scale = 25.0 / room_size
            delta_hvac = (+heater_power if HEAT, -cooling_power if COOL, 0 if OFF) * room_scale
            delta_occupancy = num_people * heat_gain_per_person * room_scale (0 if unoccupied)
            predicted_temp = current_temp + delta_ambient + delta_hvac + delta_occupancy
        """
        k = INSULATION_FACTORS.get(insulation_level, 0.07)
        delta_ambient = k * (outside_temp - current_temp)
        room_scale = 25.0 / max(float(room_size), 1.0)

        if action == ACTION_HEAT:
            delta_hvac = float(heater_power) * room_scale
        elif action == ACTION_COOL:
            delta_hvac = -float(cooling_power) * room_scale
        else:
            delta_hvac = 0.0

        effective_occupants = int(num_people) if people_present else 0
        delta_occupancy = effective_occupants * float(heat_gain_per_person) * room_scale

        return current_temp + delta_ambient + delta_hvac + delta_occupancy

    def evaluate_action(
        self,
        current_temp: float,
        desired_temp: float,
        outside_temp: float,
        insulation_level: str,
        action: str,
        heater_power: float,
        cooling_power: float,
        comfort_tolerance: float,
        people_present: bool = True,
        num_people: int = 2,
        room_size: float = 25.0,
        heat_gain_per_person: float = 0.08,
        last_action: Optional[str] = None,
        steps_in_action: int = 0,
    ) -> ActionEvaluation:
        """
        Compute the individual penalty factors and total utility for a single candidate action.
        Evaluates predicted temperature, occupancy-aware comfort penalties, energy consumption,
        overshooting risk, actuator short-cycling dwell times, and critical safety constraints.
        """
        # 1. Predict future temperature
        predicted_temp = self.predict_temperature(
            current_temp=current_temp,
            outside_temp=outside_temp,
            insulation_level=insulation_level,
            action=action,
            heater_power=heater_power,
            cooling_power=cooling_power,
            people_present=people_present,
            num_people=num_people,
            room_size=room_size,
            heat_gain_per_person=heat_gain_per_person,
        )

        effective_occupants = int(num_people) if people_present else 0
        room_scale = 25.0 / max(float(room_size), 1.0)
        delta_occupancy = effective_occupants * float(heat_gain_per_person) * room_scale
        is_occupied = bool(people_present and effective_occupants > 0)

        lower_comfort = desired_temp - comfort_tolerance
        upper_comfort = desired_temp + comfort_tolerance

        # 2. Factor 1 & Factor 4: Comfort and Overshoot Penalties
        if is_occupied:
            # Human occupants present: prioritize comfort zone around desired_temp
            diff = abs(predicted_temp - desired_temp)
            if diff <= comfort_tolerance:
                comfort_penalty = 0.5 * (diff ** 2) / max(comfort_tolerance, 0.1)
            else:
                excess = diff - comfort_tolerance
                comfort_penalty = 0.5 * comfort_tolerance + excess + (excess ** 2) / max(comfort_tolerance, 0.1)

            if predicted_temp < lower_comfort:
                overshoot_penalty = (lower_comfort - predicted_temp) * 1.5
            elif predicted_temp > upper_comfort:
                overshoot_penalty = (predicted_temp - upper_comfort) * 1.5
            else:
                overshoot_penalty = 0.0

            occupancy_comfort_factor = 1.0
        else:
            # Unoccupied room: zero human comfort penalty. All machines remain OFF to eliminate energy consumption!
            # Only critical structural/freeze safety limits (< 12°C or > 35°C) require emergency HVAC action
            if predicted_temp < self.safety_min_temp:
                excess = self.safety_min_temp - predicted_temp
                comfort_penalty = 0.5 * excess + (excess ** 2)
                overshoot_penalty = excess * 5.0 + 10.0
                occupancy_comfort_factor = 1.0
            elif predicted_temp > self.safety_max_temp:
                excess = predicted_temp - self.safety_max_temp
                comfort_penalty = 0.5 * excess + (excess ** 2)
                overshoot_penalty = excess * 5.0 + 10.0
                occupancy_comfort_factor = 1.0
            else:
                comfort_penalty = 0.0
                overshoot_penalty = 0.0
                occupancy_comfort_factor = 0.0

        # Critical Environmental Safety Constraints (overrides occupancy)
        if predicted_temp < self.safety_min_temp:
            overshoot_penalty += (self.safety_min_temp - predicted_temp) * 5.0 + 10.0
        elif predicted_temp > self.safety_max_temp:
            overshoot_penalty += (predicted_temp - self.safety_max_temp) * 5.0 + 10.0

        # 3. Factor 2: Energy Consumption Cost & Penalizing Unnecessary Energy
        base_energy = self.energy_costs.get(action, 0.0)
        energy_cost = base_energy
        if action in [ACTION_HEAT, ACTION_COOL]:
            if not is_occupied:
                # Active machines in an empty room are strictly penalized to enforce all machines OFF
                pred_off = self.predict_temperature(
                    current_temp=current_temp,
                    outside_temp=outside_temp,
                    insulation_level=insulation_level,
                    action=ACTION_OFF,
                    heater_power=heater_power,
                    cooling_power=cooling_power,
                    people_present=people_present,
                    num_people=num_people,
                    room_size=room_size,
                    heat_gain_per_person=heat_gain_per_person,
                )
                if self.safety_min_temp <= pred_off <= self.safety_max_temp:
                    energy_cost += 10.0
            else:
                # If currently maintaining minimum run duration for this active action, it is not deemed unnecessary
                is_maintaining_min_run = (last_action == action and steps_in_action < self.min_on_duration)
                if not is_maintaining_min_run:
                    pred_off = self.predict_temperature(
                        current_temp=current_temp,
                        outside_temp=outside_temp,
                        insulation_level=insulation_level,
                        action=ACTION_OFF,
                        heater_power=heater_power,
                        cooling_power=cooling_power,
                        people_present=people_present,
                        num_people=num_people,
                        room_size=room_size,
                        heat_gain_per_person=heat_gain_per_person,
                    )
                    if (lower_comfort + 0.3) <= pred_off <= (upper_comfort - 0.3):
                        energy_cost += 1.5

        # 4. Factor 3: Temperature Instability, Dwell Duration & Anti-Chattering Penalty
        temp_shift = abs(predicted_temp - current_temp)
        switch_penalty = 0.0
        if last_action is not None and action != last_action:
            # Rapid reversal between HEAT and COOL is severely penalized
            if (last_action == ACTION_HEAT and action == ACTION_COOL) or (last_action == ACTION_COOL and action == ACTION_HEAT):
                switch_penalty += 4.0
            elif last_action in [ACTION_HEAT, ACTION_COOL] and action == ACTION_OFF:
                # When room is unoccupied, shutting down all machines to eliminate energy consumption is prioritized
                if not is_occupied:
                    switch_penalty += 0.0
                elif steps_in_action < self.min_on_duration:
                    switch_penalty += 8.0
                elif is_occupied:
                    # If still significantly outside desired target, switching OFF prematurely is discouraged
                    if last_action == ACTION_HEAT and current_temp < desired_temp - 0.3:
                        switch_penalty += 1.0
                    elif last_action == ACTION_COOL and current_temp > desired_temp + 0.3:
                        switch_penalty += 1.0
            elif last_action == ACTION_OFF and action in [ACTION_HEAT, ACTION_COOL]:
                # Short-cycling: restarting equipment before minimum OFF duration
                if not is_occupied:
                    switch_penalty += 8.0
                elif steps_in_action < self.min_off_duration:
                    switch_penalty += 8.0
                elif is_occupied and lower_comfort <= current_temp <= upper_comfort:
                    switch_penalty += 1.5

        instability_penalty = temp_shift + switch_penalty

        # Weighted components
        comfort_weight_mult = occupancy_comfort_factor if is_occupied else (1.0 if comfort_penalty > 0.0 else 0.0)
        overshoot_weight_mult = 1.0

        weighted_comfort = self.weight_comfort * comfort_weight_mult * comfort_penalty
        weighted_energy = self.weight_energy * energy_cost
        weighted_stability = self.weight_stability * instability_penalty
        weighted_overshoot = self.weight_overshoot * overshoot_weight_mult * overshoot_penalty

        # Net utility: Utility(action) = - Comfort - Energy - Instability - Overshoot
        total_utility = -(
            weighted_comfort + weighted_energy + weighted_stability + weighted_overshoot
        )

        return ActionEvaluation(
            action=action,
            predicted_temp=round(predicted_temp, 2),
            comfort_penalty=round(comfort_penalty, 3),
            energy_cost=round(energy_cost, 2),
            instability_penalty=round(instability_penalty, 3),
            overshoot_penalty=round(overshoot_penalty, 3),
            weighted_comfort=round(weighted_comfort, 3),
            weighted_energy=round(weighted_energy, 3),
            weighted_stability=round(weighted_stability, 3),
            weighted_overshoot=round(weighted_overshoot, 3),
            utility=round(total_utility, 3),
            delta_occupancy=round(delta_occupancy, 3),
            room_scale=round(room_scale, 3),
            occupancy_comfort_factor=round(occupancy_comfort_factor, 2),
            is_occupied=is_occupied,
        )

    def preview_decision(self, observation: Dict[str, Any]) -> AgentDecision:
        """
        Inspect which candidate action maximizes net utility for the given observation
        without mutating energy counters or simulation history.
        """
        current_temp = float(observation["room_temp"])
        desired_temp = float(observation["desired_temp"])
        outside_temp = float(observation["outside_temp"])
        insulation_level = str(observation.get("insulation_level", "Medium"))
        heater_power = float(observation.get("heater_power", 1.5))
        cooling_power = float(observation.get("cooling_power", 1.5))
        comfort_tolerance = float(observation.get("comfort_tolerance", 1.5))
        people_present = bool(observation.get("people_present", True))
        num_people = int(observation.get("num_people", 2)) if people_present else 0
        room_size = float(observation.get("room_size", 25.0))
        heat_gain_per_person = float(observation.get("heat_gain_per_person", 0.08))
        step = int(observation.get("step", len(self.action_history)))
        prior_action = None
        steps_in_action = observation.get("steps_in_action", self.steps_in_current_action)

        if self.last_decision is not None:
            prev_step = self.last_decision.step
            prev_temp = self.last_decision.current_temp
            if step > prev_step and abs(current_temp - prev_temp) <= 3.5:
                prior_action = self.last_decision.selected_action
                if "steps_in_action" not in observation:
                    steps_in_action = self.steps_in_current_action
            else:
                steps_in_action = 0

        if "prev_action" in observation:
            prior_action = observation["prev_action"]
            if self.last_decision is not None and prior_action != self.last_decision.selected_action:
                steps_in_action = 0

        # Evaluate all candidate actions
        evaluations: Dict[str, ActionEvaluation] = {}
        for action in ALL_ACTIONS:
            evaluations[action] = self.evaluate_action(
                current_temp=current_temp,
                desired_temp=desired_temp,
                outside_temp=outside_temp,
                insulation_level=insulation_level,
                action=action,
                heater_power=heater_power,
                cooling_power=cooling_power,
                comfort_tolerance=comfort_tolerance,
                people_present=people_present,
                num_people=num_people,
                room_size=room_size,
                heat_gain_per_person=heat_gain_per_person,
                last_action=prior_action,
                steps_in_action=steps_in_action,
            )

        # Select action that maximizes net utility
        selected_action = max(evaluations.keys(), key=lambda a: evaluations[a].utility)
        selected_eval = evaluations[selected_action]

        # Formulate human-readable decision explanation
        reason = self._generate_explanation(
            selected_action=selected_action,
            selected_eval=selected_eval,
            all_evals=evaluations,
            current_temp=current_temp,
            desired_temp=desired_temp,
            outside_temp=outside_temp,
            tolerance=comfort_tolerance,
            people_present=people_present,
            num_people=num_people,
            room_size=room_size,
        )

        step_energy = self.energy_costs[selected_action]

        return AgentDecision(
            step=step,
            current_temp=current_temp,
            desired_temp=desired_temp,
            outside_temp=outside_temp,
            selected_action=selected_action,
            predicted_temp=selected_eval.predicted_temp,
            utility_score=selected_eval.utility,
            energy_consumed=step_energy,
            reason=reason,
            action_evaluations=evaluations,
            people_present=people_present,
            num_people=num_people,
            room_size=room_size,
            delta_occupancy=selected_eval.delta_occupancy,
        )

    def decide_action(self, observation: Dict[str, Any]) -> AgentDecision:
        """
        Execute the 9-step decision-making process:
        1. Observe current room temperature.
        2. Read desired temperature.
        3. Observe outside temperature and insulation.
        4. Observe occupancy status and room size.
        5. Generate possible actions (HEAT, COOL, OFF).
        6. Predict resulting temperature for each action.
        7. Calculate utility of each predicted outcome.
        8. Select action with maximum utility.
        9. Prepare applied action and energy tracking.
        10. Record and return decision summary.
        """
        decision = self.preview_decision(observation)

        if self.last_decision is not None and self.last_decision.selected_action == decision.selected_action:
            self.steps_in_current_action += 1
        else:
            self.steps_in_current_action = 1

        self.total_energy_consumed += decision.energy_consumed
        self.action_history.append(decision.selected_action)
        self.last_decision = decision

        # Log decision row
        self.decision_history.append({
            "step": decision.step,
            "room_temp": decision.current_temp,
            "outside_temp": decision.outside_temp,
            "desired_temp": decision.desired_temp,
            "action": decision.selected_action,
            "predicted_temp": decision.predicted_temp,
            "utility_score": decision.utility_score,
            "energy_consumed": decision.energy_consumed,
            "cumulative_energy": round(self.total_energy_consumed, 2),
            "reason": decision.reason,
            "people_present": decision.people_present,
            "num_people": decision.num_people,
            "room_size": decision.room_size,
            "delta_occupancy": decision.delta_occupancy,
            "u_heat": decision.action_evaluations[ACTION_HEAT].utility,
            "u_cool": decision.action_evaluations[ACTION_COOL].utility,
            "u_off": decision.action_evaluations[ACTION_OFF].utility,
        })

        return decision

    def _generate_explanation(
        self,
        selected_action: str,
        selected_eval: ActionEvaluation,
        all_evals: Dict[str, ActionEvaluation],
        current_temp: float,
        desired_temp: float,
        outside_temp: float,
        tolerance: float,
        people_present: bool = True,
        num_people: int = 2,
        room_size: float = 25.0,
    ) -> str:
        """Construct natural language justification for the selected action reflecting all factors."""
        pred = selected_eval.predicted_temp
        util = selected_eval.utility
        delta_to_target = abs(pred - desired_temp)
        delta_occ = selected_eval.delta_occupancy
        lower_comfort = desired_temp - tolerance
        upper_comfort = desired_temp + tolerance
        is_occupied = bool(people_present and num_people > 0)

        # 1. Critical safety emergency explanations
        if current_temp <= self.safety_min_temp or pred < self.safety_min_temp:
            return (
                f"Heating selected ({util:.2f}) under critical freeze safety protection: room temperature "
                f"({current_temp:.1f}°C) is at or below {self.safety_min_temp:.1f}°C, overriding standard deadband."
            )
        if current_temp >= self.safety_max_temp or pred > self.safety_max_temp:
            return (
                f"Cooling selected ({util:.2f}) under critical overheat safety protection: room temperature "
                f"({current_temp:.1f}°C) is at or above {self.safety_max_temp:.1f}°C, overriding standard deadband."
            )

        # 2. Occupancy-aware normal explanations
        if is_occupied:
            if selected_action == ACTION_HEAT:
                reason = (
                    f"Room is occupied by {num_people} person(s). Heating yields highest utility ({util:.2f}) "
                    f"in the {room_size:.0f}m² room (+{delta_occ:.2f}°C body heat), warming temperature to {pred:.1f}°C "
                    f"within the [{lower_comfort:.1f}°C–{upper_comfort:.1f}°C] comfort deadband toward setpoint {desired_temp:.1f}°C."
                )
            elif selected_action == ACTION_COOL:
                reason = (
                    f"Room is occupied by {num_people} person(s). Cooling yields highest utility ({util:.2f}) "
                    f"by counteracting internal metabolic heat (+{delta_occ:.2f}°C) and outside warmth ({outside_temp:.1f}°C) "
                    f"to maintain temperature ({pred:.1f}°C) within the [{lower_comfort:.1f}°C–{upper_comfort:.1f}°C] comfort deadband."
                )
            else:  # ACTION_OFF
                if delta_to_target <= tolerance:
                    reason = (
                        f"Room is occupied by {num_people} person(s). HVAC remains OFF ({util:.2f}) because the {room_size:.0f}m² "
                        f"room naturally stays within the [{lower_comfort:.1f}°C–{upper_comfort:.1f}°C] comfort deadband (predicted {pred:.1f}°C), "
                        f"eliminating active energy consumption."
                    )
                else:
                    reason = (
                        f"Room is occupied by {num_people} person(s). HVAC remains OFF ({util:.2f}) as running active heating/cooling "
                        f"would incur energy and stability penalties exceeding minor comfort improvements."
                    )
        else:
            # Unoccupied room: all machines remain OFF to maximize energy conservation
            if selected_action == ACTION_OFF:
                reason = (
                    f"All machines remain OFF because the room is unoccupied (0 occupants). "
                    f"Temperature ({current_temp:.1f}°C → {pred:.1f}°C) is safe within [{self.safety_min_temp:.1f}°C–{self.safety_max_temp:.1f}°C]. "
                    f"HVAC equipment and ceiling fans are powered OFF for total energy conservation (energy cost: 0, utility: {util:.2f})."
                )
            elif selected_action == ACTION_COOL:
                reason = (
                    f"Room is unoccupied (0 occupants), but emergency cooling is activated (utility: {util:.2f}) because temperature "
                    f"({current_temp:.1f}°C) exceeded the critical safety ceiling ({self.safety_max_temp:.1f}°C), protecting building and electronics."
                )
            else:  # ACTION_HEAT
                reason = (
                    f"Room is unoccupied (0 occupants), but emergency heating is activated (utility: {util:.2f}) because temperature "
                    f"({current_temp:.1f}°C) fell below the critical freeze safety floor ({self.safety_min_temp:.1f}°C), preventing freeze damage."
                )

        return reason

    def get_action_scores_dict(self) -> Dict[str, float]:
        """Return simple action-to-utility mapping for charting."""
        if not self.last_decision:
            return {a: 0.0 for a in ALL_ACTIONS}
        return {a: self.last_decision.action_evaluations[a].utility for a in ALL_ACTIONS}

    def get_decision_history_df(self) -> pd.DataFrame:
        """Return the complete log of agent decisions as a pandas DataFrame."""
        return pd.DataFrame(self.decision_history)

    def get_metrics(self) -> Dict[str, Any]:
        """Return cumulative performance metrics."""
        total_steps = len(self.action_history)
        heat_steps = self.action_history.count(ACTION_HEAT)
        cool_steps = self.action_history.count(ACTION_COOL)
        off_steps = self.action_history.count(ACTION_OFF)

        return {
            "total_steps": total_steps,
            "total_energy": round(self.total_energy_consumed, 1),
            "heat_percentage": round((heat_steps / total_steps * 100) if total_steps else 0, 1),
            "cool_percentage": round((cool_steps / total_steps * 100) if total_steps else 0, 1),
            "off_percentage": round((off_steps / total_steps * 100) if total_steps else 0, 1),
            "last_action": self.last_decision.selected_action if self.last_decision else ACTION_OFF,
            "last_utility": self.last_decision.utility_score if self.last_decision else 0.0,
        }

