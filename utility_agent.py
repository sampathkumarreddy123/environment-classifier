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
    ) -> None:
        """
        Initialize the utility-based agent with configurable weights and costs.

        Args:
            weight_comfort: Importance weight for proximity to target temperature.
            weight_energy: Importance weight for minimizing power consumption.
            weight_stability: Importance weight for minimizing abrupt temperature shifts.
            weight_overshoot: Importance weight for avoiding comfort zone violations.
            energy_cost_heat: Energy cost unit for operating the heater.
            energy_cost_cool: Energy cost unit for operating the cooler.
            energy_cost_off: Energy cost unit when HVAC is disabled.
        """
        self.weight_comfort = float(weight_comfort)
        self.weight_energy = float(weight_energy)
        self.weight_stability = float(weight_stability)
        self.weight_overshoot = float(weight_overshoot)

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

    def reset(self) -> None:
        """Reset agent memory and cumulative metrics."""
        self.last_decision = None
        self.total_energy_consumed = 0.0
        self.action_history = []
        self.decision_history = []

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
    ) -> float:
        """
        Predict the next room temperature for a candidate action using the
        physical thermodynamic model of the environment.

        Model:
            delta_ambient = k * (outside_temp - current_temp)
            delta_hvac = +heater_power (if HEAT), -cooling_power (if COOL), 0 (if OFF)
            predicted_temp = current_temp + delta_ambient + delta_hvac
        """
        k = INSULATION_FACTORS.get(insulation_level, 0.07)
        delta_ambient = k * (outside_temp - current_temp)

        if action == ACTION_HEAT:
            delta_hvac = heater_power
        elif action == ACTION_COOL:
            delta_hvac = -cooling_power
        else:
            delta_hvac = 0.0

        return current_temp + delta_ambient + delta_hvac

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
    ) -> ActionEvaluation:
        """
        Compute the individual penalty factors and total utility for a single candidate action.
        """
        # 1. Predict future temperature
        predicted_temp = self.predict_temperature(
            current_temp=current_temp,
            outside_temp=outside_temp,
            insulation_level=insulation_level,
            action=action,
            heater_power=heater_power,
            cooling_power=cooling_power,
        )

        # 2. Factor 1: Temperature Comfort Penalty (|predicted - desired|)
        comfort_penalty = abs(predicted_temp - desired_temp)

        # 3. Factor 2: Energy Consumption Cost
        energy_cost = self.energy_costs.get(action, 0.0)

        # 4. Factor 3: Temperature Instability Penalty (|predicted - current|)
        instability_penalty = abs(predicted_temp - current_temp)

        # 5. Factor 4: Overshooting Penalty (distance beyond comfort zone boundaries)
        lower_comfort = desired_temp - comfort_tolerance
        upper_comfort = desired_temp + comfort_tolerance
        if predicted_temp < lower_comfort:
            overshoot_penalty = lower_comfort - predicted_temp
        elif predicted_temp > upper_comfort:
            overshoot_penalty = predicted_temp - upper_comfort
        else:
            overshoot_penalty = 0.0

        # Weighted components
        weighted_comfort = self.weight_comfort * comfort_penalty
        weighted_energy = self.weight_energy * energy_cost
        weighted_stability = self.weight_stability * instability_penalty
        weighted_overshoot = self.weight_overshoot * overshoot_penalty

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
        )

    def decide_action(self, observation: Dict[str, Any]) -> AgentDecision:
        """
        Execute the 9-step decision-making process:
        1. Observe current room temperature.
        2. Read desired temperature.
        3. Observe outside temperature and insulation.
        4. Generate possible actions (HEAT, COOL, OFF).
        5. Predict resulting temperature for each action.
        6. Calculate utility of each predicted outcome.
        7. Select action with maximum utility.
        8. Prepare applied action and energy tracking.
        9. Record and return decision summary.
        """
        current_temp = float(observation["room_temp"])
        desired_temp = float(observation["desired_temp"])
        outside_temp = float(observation["outside_temp"])
        insulation_level = str(observation.get("insulation_level", "Medium"))
        heater_power = float(observation.get("heater_power", 1.5))
        cooling_power = float(observation.get("cooling_power", 1.5))
        comfort_tolerance = float(observation.get("comfort_tolerance", 1.5))
        step = int(observation.get("step", len(self.action_history)))

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
        )

        step_energy = self.energy_costs[selected_action]
        self.total_energy_consumed += step_energy
        self.action_history.append(selected_action)

        decision = AgentDecision(
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
        )
        self.last_decision = decision

        # Log decision row
        self.decision_history.append({
            "step": step,
            "room_temp": current_temp,
            "outside_temp": outside_temp,
            "desired_temp": desired_temp,
            "action": selected_action,
            "predicted_temp": selected_eval.predicted_temp,
            "utility_score": selected_eval.utility,
            "energy_consumed": step_energy,
            "cumulative_energy": round(self.total_energy_consumed, 2),
            "reason": reason,
            "u_heat": evaluations[ACTION_HEAT].utility,
            "u_cool": evaluations[ACTION_COOL].utility,
            "u_off": evaluations[ACTION_OFF].utility,
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
    ) -> str:
        """Construct natural language justification for the selected action."""
        pred = selected_eval.predicted_temp
        util = selected_eval.utility
        delta_to_target = abs(pred - desired_temp)

        if selected_action == ACTION_HEAT:
            reason = (
                f"Heating yields highest utility ({util:.2f}) by warming room to {pred:.1f}°C "
                f"(within {delta_to_target:.1f}°C of target {desired_temp:.1f}°C), outweighing "
                f"the energy cost ({selected_eval.weighted_energy:.2f})."
            )
        elif selected_action == ACTION_COOL:
            reason = (
                f"Cooling yields highest utility ({util:.2f}) by lowering temperature to {pred:.1f}°C "
                f"(within {delta_to_target:.1f}°C of target {desired_temp:.1f}°C), offsetting the cooling power cost."
            )
        else:  # ACTION_OFF
            if delta_to_target <= tolerance:
                reason = (
                    f"HVAC OFF yields highest utility ({util:.2f}) because the room naturally remains "
                    f"comfortable ({pred:.1f}°C), eliminating active energy consumption (cost: 0)."
                )
            else:
                reason = (
                    f"HVAC OFF selected ({util:.2f}) as running active heating/cooling would incur high "
                    f"energy and stability penalties that outweigh minor comfort adjustments."
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

