"""
Intelligent Agents for the Room Temperature Control Simulation.

Implements:
1. SimpleReflexAgent: Condition-action rule thermostat using fixed temperature thresholds.
2. UtilityBasedAgent: Decision-making agent that projects thermal outcomes and maximizes
   a multi-objective utility function balancing comfort, energy consumption, and actuator switching.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np

from config import (
    ACTION_COOL,
    ACTION_HEAT,
    ACTION_OFF,
    AGENT_REFLEX,
    AGENT_UTILITY,
    ALL_ACTIONS,
    AgentDefaults,
    INSULATION_FACTORS,
)


class BaseAgent(ABC):
    """Abstract base class for temperature control agents."""

    def __init__(self, name: str, agent_type: str) -> None:
        self.name = name
        self.agent_type = agent_type
        self.last_action = ACTION_OFF
        self.total_energy_consumed = 0.0
        self.switch_count = 0
        self.action_history: List[str] = []
        self.last_reasoning: str = ""

    def reset(self) -> None:
        """Reset agent memory and metrics."""
        self.last_action = ACTION_OFF
        self.total_energy_consumed = 0.0
        self.switch_count = 0
        self.action_history = []
        self.last_reasoning = ""

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> str:
        """
        Determine the next action given the current environment observation.

        Args:
            observation: State dictionary from RoomEnvironment.

        Returns:
            Selected action: 'OFF', 'HEAT', or 'COOL'.
        """
        pass

    def _track_action(self, action: str) -> None:
        """Track action metrics, energy costs, and actuator switching."""
        if action != self.last_action and self.last_action != ACTION_OFF:
            self.switch_count += 1

        cost = (
            AgentDefaults.ENERGY_COST_HEAT
            if action == ACTION_HEAT
            else AgentDefaults.ENERGY_COST_COOL
            if action == ACTION_COOL
            else AgentDefaults.ENERGY_COST_OFF
        )
        self.total_energy_consumed += cost
        self.last_action = action
        self.action_history.append(action)

    def get_metrics(self) -> Dict[str, Any]:
        """Return cumulative performance metrics for this agent."""
        total_steps = len(self.action_history)
        heat_steps = self.action_history.count(ACTION_HEAT)
        cool_steps = self.action_history.count(ACTION_COOL)
        off_steps = self.action_history.count(ACTION_OFF)

        return {
            "name": self.name,
            "type": self.agent_type,
            "total_steps": total_steps,
            "total_energy": round(self.total_energy_consumed, 1),
            "switch_count": self.switch_count,
            "heat_percentage": round((heat_steps / total_steps * 100) if total_steps else 0, 1),
            "cool_percentage": round((cool_steps / total_steps * 100) if total_steps else 0, 1),
            "off_percentage": round((off_steps / total_steps * 100) if total_steps else 0, 1),
        }


class SimpleReflexAgent(BaseAgent):
    """
    A simple reflex thermostat agent that makes immediate decisions using fixed temperature thresholds.

    Rules:
    - If Current Temp < Desired Temp - Comfort Tolerance: Turn on HEATER.
    - If Current Temp > Desired Temp + Comfort Tolerance: Turn on COOLER.
    - Otherwise (Within Comfort Zone): Turn OFF.
    """

    def __init__(
        self,
        name: str = AGENT_REFLEX,
        hysteresis: float = AgentDefaults.DEFAULT_REFLEX_HYSTERESIS,
    ) -> None:
        super().__init__(name=name, agent_type=AGENT_REFLEX)
        self.hysteresis = float(hysteresis)

    def act(self, observation: Dict[str, Any]) -> str:
        current_temp = float(observation["room_temp"])
        desired_temp = float(observation["desired_temp"])
        tolerance = float(observation.get("comfort_tolerance", 1.5))

        lower_threshold = desired_temp - tolerance - self.hysteresis
        upper_threshold = desired_temp + tolerance + self.hysteresis

        # Rule evaluation
        if current_temp < lower_threshold:
            action = ACTION_HEAT
            self.last_reasoning = (
                f"Room temp ({current_temp:.2f}°C) is below lower comfort bound ({lower_threshold:.2f}°C). "
                "Triggering HEATER."
            )
        elif current_temp > upper_threshold:
            action = ACTION_COOL
            self.last_reasoning = (
                f"Room temp ({current_temp:.2f}°C) is above upper comfort bound ({upper_threshold:.2f}°C). "
                "Triggering COOLER."
            )
        else:
            action = ACTION_OFF
            self.last_reasoning = (
                f"Room temp ({current_temp:.2f}°C) is within comfortable range "
                f"[{lower_threshold:.2f}°C, {upper_threshold:.2f}°C]. HVAC remains OFF."
            )

        self._track_action(action)
        return action


class UtilityBasedAgent(BaseAgent):
    """
    A utility-based agent that projects next-step thermal outcomes for all candidate actions
    and selects the action maximizing an explicit multi-objective utility function:

        Utility(a) = U_comfort(T_next) + U_energy(a) + U_switch(a)

    Where:
    - U_comfort rewards proximity to the desired temperature and penalizes deviation.
    - U_energy penalizes electrical power consumption.
    - U_switch penalizes frequent cycling of the heating/cooling compressors.
    """

    def __init__(
        self,
        name: str = AGENT_UTILITY,
        weight_comfort: float = AgentDefaults.DEFAULT_WEIGHT_COMFORT,
        weight_energy: float = AgentDefaults.DEFAULT_WEIGHT_ENERGY,
        weight_switching: float = AgentDefaults.DEFAULT_WEIGHT_SWITCHING,
    ) -> None:
        super().__init__(name=name, agent_type=AGENT_UTILITY)
        self.weight_comfort = float(weight_comfort)
        self.weight_energy = float(weight_energy)
        self.weight_switching = float(weight_switching)
        self.last_action_utilities: Dict[str, Dict[str, float]] = {}

    def act(self, observation: Dict[str, Any]) -> str:
        current_temp = float(observation["room_temp"])
        desired_temp = float(observation["desired_temp"])
        outside_temp = float(observation["outside_temp"])
        tolerance = float(observation.get("comfort_tolerance", 1.5))
        insulation_level = observation.get("insulation_level", "Medium")
        heater_power = float(observation.get("heater_power", 1.5))
        cooling_power = float(observation.get("cooling_power", 1.5))

        k_loss = INSULATION_FACTORS.get(insulation_level, 0.07)
        delta_ambient = k_loss * (outside_temp - current_temp)

        best_action = ACTION_OFF
        max_utility = -float("inf")
        self.last_action_utilities = {}

        for action in ALL_ACTIONS:
            # 1. Project next temperature for candidate action (internal mental model)
            if action == ACTION_HEAT:
                delta_hvac = heater_power
                energy_cost = AgentDefaults.ENERGY_COST_HEAT
            elif action == ACTION_COOL:
                delta_hvac = -cooling_power
                energy_cost = AgentDefaults.ENERGY_COST_COOL
            else:
                delta_hvac = 0.0
                energy_cost = AgentDefaults.ENERGY_COST_OFF

            projected_temp = current_temp + delta_ambient + delta_hvac
            temp_error = abs(projected_temp - desired_temp)
            is_in_comfort = temp_error <= tolerance

            # 2. Compute Utility Components
            # Comfort score: quadratic penalty scaled by tolerance + comfort bonus
            normalized_error = temp_error / max(tolerance, 0.1)
            u_comfort = -self.weight_comfort * (normalized_error ** 2)
            if is_in_comfort:
                u_comfort += AgentDefaults.COMFORT_BONUS

            # Energy score: penalty for consumption
            u_energy = -self.weight_energy * energy_cost

            # Actuator switching penalty: penalize frequent toggles
            is_switch = (action != self.last_action) and (self.last_action != ACTION_OFF)
            u_switch = -self.weight_switching if is_switch else 0.0

            total_utility = u_comfort + u_energy + u_switch

            self.last_action_utilities[action] = {
                "projected_temp": round(projected_temp, 2),
                "u_comfort": round(u_comfort, 2),
                "u_energy": round(u_energy, 2),
                "u_switch": round(u_switch, 2),
                "total_utility": round(total_utility, 3),
            }

            if total_utility > max_utility:
                max_utility = total_utility
                best_action = action

        best_meta = self.last_action_utilities[best_action]
        self.last_reasoning = (
            f"Evaluated all candidate actions. Selected {best_action} with maximum utility "
            f"({best_meta['total_utility']:.2f}). Projected Temp: {best_meta['projected_temp']:.2f}°C "
            f"[Comfort: {best_meta['u_comfort']}, Energy: {best_meta['u_energy']}, Switch: {best_meta['u_switch']}]."
        )

        self._track_action(best_action)
        return best_action

