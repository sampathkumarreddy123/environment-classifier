"""
Configuration settings and default parameters for the Environment Classifier
and Demonstrator simulation.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SimulationDefaults:
    """Default parameters and constraints for the room environment simulation."""
    
    # Temperature limits (°C)
    MIN_INITIAL_TEMP: float = 10.0
    MAX_INITIAL_TEMP: float = 40.0
    DEFAULT_INITIAL_TEMP: float = 18.0

    MIN_DESIRED_TEMP: float = 16.0
    MAX_DESIRED_TEMP: float = 30.0
    DEFAULT_DESIRED_TEMP: float = 22.0

    MIN_OUTSIDE_TEMP: float = 0.0
    MAX_OUTSIDE_TEMP: float = 45.0
    DEFAULT_OUTSIDE_TEMP: float = 34.0

    # Comfort range tolerance around desired temperature (± °C)
    DEFAULT_COMFORT_TOLERANCE: float = 1.5

    # Simulation steps
    MIN_STEPS: int = 10
    MAX_STEPS: int = 200
    DEFAULT_STEPS: int = 50

    # HVAC Power (°C change potential per time step)
    MIN_HEATER_POWER: float = 0.2
    MAX_HEATER_POWER: float = 4.0
    DEFAULT_HEATER_POWER: float = 1.5

    MIN_COOLING_POWER: float = 0.2
    MAX_COOLING_POWER: float = 4.0
    DEFAULT_COOLING_POWER: float = 1.5

    # Environmental disturbance standard deviation (°C)
    MIN_DISTURBANCE: float = 0.0
    MAX_DISTURBANCE: float = 2.0
    DEFAULT_DISTURBANCE: float = 0.2


@dataclass(frozen=True)
class AgentDefaults:
    """Default parameters for Simple Reflex and Utility-Based agents."""

    # Reflex Agent parameters
    DEFAULT_REFLEX_HYSTERESIS: float = 0.0  # Optional deadband offset

    # Utility-Based Agent weights and costs
    DEFAULT_WEIGHT_COMFORT: float = 1.0     # Multiplier for comfort penalty / reward
    DEFAULT_WEIGHT_ENERGY: float = 0.4      # Multiplier for energy cost penalty
    DEFAULT_WEIGHT_SWITCHING: float = 0.1   # Penalty for toggling HVAC state

    # Energy consumption units per action
    ENERGY_COST_OFF: float = 0.0
    ENERGY_COST_HEAT: float = 1.0
    ENERGY_COST_COOL: float = 1.0

    # Comfort bonus when within target tolerance
    COMFORT_BONUS: float = 2.0


# Insulation levels mapped to heat transfer coefficients (k)
# Higher k means faster heat transfer through walls/windows
INSULATION_FACTORS: Dict[str, float] = {
    "Poor": 0.14,      # e.g., single pane windows, no wall insulation
    "Medium": 0.07,    # e.g., standard double glazed, moderate insulation
    "Good": 0.025,     # e.g., modern well-sealed building, high-grade insulation
}

# Temperature Classifications
CLASS_COLD = "Cold"
CLASS_COMFORTABLE = "Comfortable"
CLASS_HOT = "Hot"

# Available HVAC actions
ACTION_OFF = "OFF"
ACTION_HEAT = "HEAT"
ACTION_COOL = "COOL"
ALL_ACTIONS = [ACTION_OFF, ACTION_HEAT, ACTION_COOL]

# Agent Identifiers
AGENT_REFLEX = "Simple Reflex Thermostat"
AGENT_UTILITY = "Utility-Based Agent"
ALL_AGENTS = [AGENT_REFLEX, AGENT_UTILITY]

# Classification Color Scheme (Clean, modern palette)
STATUS_COLORS = {
    CLASS_COLD: {
        "color": "#0284c7",       # Sky blue
        "bg_light": "#e0f2fe",
        "border": "#7dd3fc",
        "icon": "❄️",
        "description": "Temperature is below the comfortable range.",
    },
    CLASS_COMFORTABLE: {
        "color": "#16a34a",       # Fresh green
        "bg_light": "#dcfce7",
        "border": "#86efac",
        "icon": "🌿",
        "description": "Temperature is within the optimal comfort zone.",
    },
    CLASS_HOT: {
        "color": "#dc2626",       # Vibrant red
        "bg_light": "#fee2e2",
        "border": "#fca5a5",
        "icon": "🔥",
        "description": "Temperature is above the comfortable range.",
    },
}
