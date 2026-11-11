"""
Configuration settings and default parameters for the Environment Classifier
and Demonstrator simulation.
"""

from dataclasses import dataclass
from typing import Dict, Optional


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
class UtilityWeightsDefaults:
    """Default configurable weights for the Utility-Based Agent."""
    
    # Weight factors for utility function penalties
    DEFAULT_WEIGHT_COMFORT: float = 1.0     # Multiplier for comfort deviation
    DEFAULT_WEIGHT_ENERGY: float = 0.5      # Multiplier for energy consumption
    DEFAULT_WEIGHT_STABILITY: float = 0.2   # Multiplier for temperature change magnitude
    DEFAULT_WEIGHT_OVERSHOOT: float = 1.0   # Multiplier for comfort zone bounds violations

    # Energy consumption costs per action
    ENERGY_COST_HEAT: float = 1.0
    ENERGY_COST_COOL: float = 1.0
    ENERGY_COST_OFF: float = 0.0


@dataclass(frozen=True)
class ReflexThermostatDefaults:
    """Default configurable thresholds and energy costs for the Simple Reflex Thermostat."""

    MIN_THRESHOLD: float = 10.0
    MAX_THRESHOLD: float = 35.0
    DEFAULT_LOWER_THRESHOLD: float = 20.0
    DEFAULT_UPPER_THRESHOLD: float = 26.0
    THRESHOLD_STEP: float = 0.5

    # Energy consumption costs per action
    ENERGY_COST_HEAT: float = 1.0
    ENERGY_COST_COOL: float = 1.0
    ENERGY_COST_OFF: float = 0.0


@dataclass(frozen=True)
class DemonstrationScenario:
    """Preconfigured demonstration scenario for academic evaluation."""
    id: str
    name: str
    tag: str
    description: str
    initial_temp: float
    desired_temp: float
    outside_temp: float
    insulation_level: str
    comfort_tolerance: float
    heater_power: float
    cooling_power: float
    disturbance_level: float
    max_steps: int
    random_seed: int
    lower_threshold: float
    upper_threshold: float
    weight_comfort: float
    weight_energy: float
    weight_stability: float
    weight_overshoot: float


# Preconfigured academic demonstration scenarios
DEMO_SCENARIOS: Dict[str, DemonstrationScenario] = {
    "cold_room": DemonstrationScenario(
        id="cold_room",
        name="Scenario 1: Cold Room",
        tag="Heating Response",
        description="Initial room temperature (14.0°C) starts well below target (22.0°C) during chilly outdoor conditions (10.0°C). Demonstrates how both agents heat the room, recover comfort, and prevent thermal overshoot.",
        initial_temp=14.0,
        desired_temp=22.0,
        outside_temp=10.0,
        insulation_level="Medium",
        comfort_tolerance=1.5,
        heater_power=1.5,
        cooling_power=1.5,
        disturbance_level=0.2,
        max_steps=50,
        random_seed=42,
        lower_threshold=20.5,
        upper_threshold=23.5,
        weight_comfort=1.0,
        weight_energy=0.5,
        weight_stability=0.2,
        weight_overshoot=1.0,
    ),
    "hot_room": DemonstrationScenario(
        id="hot_room",
        name="Scenario 2: Hot Room",
        tag="Cooling Response",
        description="Initial room temperature (32.0°C) starts well above target (22.0°C) during a hot summer heatwave (36.0°C). Demonstrates cooling actuation, energy consumption, and settling stability.",
        initial_temp=32.0,
        desired_temp=22.0,
        outside_temp=36.0,
        insulation_level="Medium",
        comfort_tolerance=1.5,
        heater_power=1.5,
        cooling_power=1.5,
        disturbance_level=0.2,
        max_steps=50,
        random_seed=42,
        lower_threshold=20.5,
        upper_threshold=23.5,
        weight_comfort=1.0,
        weight_energy=0.5,
        weight_stability=0.2,
        weight_overshoot=1.0,
    ),
    "comfortable_room": DemonstrationScenario(
        id="comfortable_room",
        name="Scenario 3: Comfortable Room",
        tag="Comfort Maintenance & Efficiency",
        description="Room starts at 22.0°C (inside the comfort target zone) with mild outdoor conditions (28.0°C) and good insulation. Demonstrates how agents minimize active energy while maintaining thermal stability.",
        initial_temp=22.0,
        desired_temp=22.0,
        outside_temp=28.0,
        insulation_level="Good",
        comfort_tolerance=1.5,
        heater_power=1.5,
        cooling_power=1.5,
        disturbance_level=0.15,
        max_steps=50,
        random_seed=42,
        lower_threshold=20.5,
        upper_threshold=23.5,
        weight_comfort=1.0,
        weight_energy=0.5,
        weight_stability=0.2,
        weight_overshoot=1.0,
    ),
}


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
ALL_ACTIONS = [ACTION_HEAT, ACTION_COOL, ACTION_OFF]

# Classification Color Scheme (Biophilic Eco-Tech Palette)
STATUS_COLORS = {
    CLASS_COLD: {
        "color": "#4D9285",       # Muted Teal / Crisp Glacial
        "bg_light": "#EAF0E5",
        "border": "#DCE5D8",
        "icon": "❄️",
        "description": "Temperature is below the comfortable range.",
    },
    CLASS_COMFORTABLE: {
        "color": "#39845A",       # Natural Success Green
        "bg_light": "#EAF0E5",
        "border": "#829B78",
        "icon": "🌿",
        "description": "Temperature is within the optimal comfort zone.",
    },
    CLASS_HOT: {
        "color": "#C65B52",       # Earthy Terracotta / Heat
        "bg_light": "#FDF2F0",
        "border": "#E58B80",
        "icon": "🔥",
        "description": "Temperature is above the comfortable range.",
    },
}

# Action Display Metadata
ACTION_METADATA = {
    ACTION_HEAT: {
        "name": "HEAT",
        "label": "🔥 HEAT",
        "color": "#C65B52",
        "description": "Engage heater (+Power)",
    },
    ACTION_COOL: {
        "name": "COOL",
        "label": "❄️ COOL",
        "color": "#4D9285",
        "description": "Engage air conditioner (-Power)",
    },
    ACTION_OFF: {
        "name": "OFF",
        "label": "⏸️ OFF",
        "color": "#5F7162",
        "description": "Disable HVAC (Passive drift)",
    },
}
