"""
Room Environment Simulation Model.

Implements a simplified thermodynamic heat transfer model of a room,
accounting for outdoor temperature, wall insulation, HVAC heating/cooling,
and stochastic environmental disturbances.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from config import (
    ACTION_COOL,
    ACTION_HEAT,
    ACTION_OFF,
    CLASS_COLD,
    CLASS_COMFORTABLE,
    CLASS_HOT,
    INSULATION_FACTORS,
    SimulationDefaults,
)


class RoomEnvironment:
    """Simulates a continuous room environment with thermodynamic heat exchange."""

    def __init__(
        self,
        initial_temp: float = SimulationDefaults.DEFAULT_INITIAL_TEMP,
        desired_temp: float = SimulationDefaults.DEFAULT_DESIRED_TEMP,
        outside_temp: float = SimulationDefaults.DEFAULT_OUTSIDE_TEMP,
        insulation_level: str = "Medium",
        heater_power: float = SimulationDefaults.DEFAULT_HEATER_POWER,
        cooling_power: float = SimulationDefaults.DEFAULT_COOLING_POWER,
        disturbance_level: float = SimulationDefaults.DEFAULT_DISTURBANCE,
        comfort_tolerance: float = SimulationDefaults.DEFAULT_COMFORT_TOLERANCE,
        random_seed: Optional[int] = None,
        people_present: bool = False,
        num_people: int = 0,
        room_size: float = SimulationDefaults.DEFAULT_ROOM_SIZE,
        heat_gain_per_person: float = SimulationDefaults.DEFAULT_HEAT_GAIN_PER_PERSON,
    ) -> None:
        """
        Initialize the room simulation environment.

        Args:
            initial_temp: Starting indoor temperature (°C).
            desired_temp: Target comfort temperature (°C).
            outside_temp: External ambient temperature (°C).
            insulation_level: Insulation quality ('Poor', 'Medium', 'Good').
            heater_power: Maximum temperature rise per step when heating (°C).
            cooling_power: Maximum temperature drop per step when cooling (°C).
            disturbance_level: Standard deviation of stochastic noise (°C).
            comfort_tolerance: Half-width of comfort range around desired_temp (°C).
            random_seed: Optional random seed for reproducible disturbance sampling.
            people_present: Whether occupants are present in the room (bool).
            num_people: Number of people inside the room (int).
            room_size: Room floor area in m² (float).
            heat_gain_per_person: Metabolic heat gain per person per step at reference size (°C).
        """
        self.initial_temp = float(initial_temp)
        self.desired_temp = float(desired_temp)
        self.outside_temp = float(outside_temp)
        self.insulation_level = insulation_level
        self.heater_power = float(heater_power)
        self.cooling_power = float(cooling_power)
        self.disturbance_level = float(disturbance_level)
        self.comfort_tolerance = float(comfort_tolerance)
        self.random_seed = random_seed

        # Occupancy & Room configuration
        self.people_present = bool(people_present)
        self.num_people = int(num_people) if self.people_present else 0
        self.room_size = float(room_size)
        self.heat_gain_per_person = float(heat_gain_per_person)
        # Thermal mass scaling: room_scale = A_ref / A_room (Volume = Area * 2.5m)
        self.room_scale = SimulationDefaults.REFERENCE_ROOM_SIZE / max(self.room_size, 1.0)

        self._rng = np.random.default_rng(random_seed)

        # Heat transfer rate factor (Newton's law of cooling coefficient)
        self.k_loss = INSULATION_FACTORS.get(self.insulation_level, 0.07)

        # Simulation state tracking
        self.current_step = 0
        self.current_temp = self.initial_temp
        self.history: List[Dict[str, Any]] = []

        # Record initial baseline state at step 0
        self._record_step(
            step=0,
            temp=self.current_temp,
            action=ACTION_OFF,
            delta_ambient=0.0,
            delta_hvac=0.0,
            delta_occupancy=0.0,
            disturbance=0.0,
        )

    def reset(
        self,
        new_initial_temp: Optional[float] = None,
        new_desired_temp: Optional[float] = None,
        new_outside_temp: Optional[float] = None,
        new_insulation_level: Optional[str] = None,
        new_heater_power: Optional[float] = None,
        new_cooling_power: Optional[float] = None,
        new_disturbance_level: Optional[float] = None,
        new_people_present: Optional[bool] = None,
        new_num_people: Optional[int] = None,
        new_room_size: Optional[float] = None,
        new_heat_gain_per_person: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Reset the environment to time step 0, optionally updating configuration.

        Returns:
            Observation dictionary of the initial state.
        """
        if new_initial_temp is not None:
            self.initial_temp = float(new_initial_temp)
        if new_desired_temp is not None:
            self.desired_temp = float(new_desired_temp)
        if new_outside_temp is not None:
            self.outside_temp = float(new_outside_temp)
        if new_insulation_level is not None:
            self.insulation_level = new_insulation_level
            self.k_loss = INSULATION_FACTORS.get(self.insulation_level, 0.07)
        if new_heater_power is not None:
            self.heater_power = float(new_heater_power)
        if new_cooling_power is not None:
            self.cooling_power = float(new_cooling_power)
        if new_disturbance_level is not None:
            self.disturbance_level = float(new_disturbance_level)
        if new_people_present is not None:
            self.people_present = bool(new_people_present)
        if new_num_people is not None:
            self.num_people = int(new_num_people) if self.people_present else 0
        if not self.people_present:
            self.num_people = 0
        if new_room_size is not None:
            self.room_size = float(new_room_size)
            self.room_scale = SimulationDefaults.REFERENCE_ROOM_SIZE / max(self.room_size, 1.0)
        if new_heat_gain_per_person is not None:
            self.heat_gain_per_person = float(new_heat_gain_per_person)

        if self.random_seed is not None:
            self._rng = np.random.default_rng(self.random_seed)

        self.current_step = 0
        self.current_temp = self.initial_temp
        self.history = []

        self._record_step(
            step=0,
            temp=self.current_temp,
            action=ACTION_OFF,
            delta_ambient=0.0,
            delta_hvac=0.0,
            delta_occupancy=0.0,
            disturbance=0.0,
        )

        return self.get_state()

    def classify_temperature(self, temp: Optional[float] = None) -> str:
        """
        Classify a temperature relative to the desired temperature comfort band.

        Comfort range: [desired - tolerance, desired + tolerance]

        Args:
            temp: Temperature to classify. Defaults to current_temp.

        Returns:
            'Cold', 'Comfortable', or 'Hot'.
        """
        t = self.current_temp if temp is None else float(temp)
        lower_bound = self.desired_temp - self.comfort_tolerance
        upper_bound = self.desired_temp + self.comfort_tolerance

        if t < lower_bound:
            return CLASS_COLD
        elif t > upper_bound:
            return CLASS_HOT
        else:
            return CLASS_COMFORTABLE

    @property
    def current_classification(self) -> str:
        """Return the classification of current room temperature."""
        return self.classify_temperature(self.current_temp)

    @property
    def is_comfortable(self) -> bool:
        """Return True if current temperature is within comfortable range."""
        return self.current_classification == CLASS_COMFORTABLE

    @property
    def delta_occupancy(self) -> float:
        """Sensible metabolic body heat added per step under current occupancy and room scale (°C)."""
        if not self.people_present:
            return 0.0
        return float(self.num_people) * self.heat_gain_per_person * self.room_scale

    @property
    def delta_hvac_heat(self) -> float:
        """Heating thermal response scaled by room thermal mass (°C)."""
        return self.heater_power * self.room_scale

    @property
    def delta_hvac_cool(self) -> float:
        """Cooling thermal response scaled by room thermal mass (°C)."""
        return self.cooling_power * self.room_scale

    def step(
        self,
        action: str = ACTION_OFF,
        disturbance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Advance the simulation by one discrete time step.

        Thermodynamic Model:
            1. Ambient Heat Transfer (Newton's cooling law):
               delta_ambient = k_loss * (outside_temp - current_temp)

            2. HVAC Action:
               - 'HEAT': +heater_power
               - 'COOL': -cooling_power
               - 'OFF': 0.0

            3. Stochastic Disturbance:
               noise ~ N(0, disturbance_level^2) (or explicit shared disturbance)

            Total delta_T = delta_ambient + delta_hvac + disturbance
            current_temp_{t+1} = current_temp_t + delta_T

        Args:
            action: Action taken ('OFF', 'HEAT', 'COOL').
            disturbance: Optional externally supplied disturbance (°C) for
                synchronized multi-agent comparison runs.

        Returns:
            Dictionary containing observation of the new state.
        """
        action = action.upper()
        if action not in [ACTION_OFF, ACTION_HEAT, ACTION_COOL]:
            action = ACTION_OFF

        # 1. Passive heat exchange through walls/windows
        delta_ambient = self.k_loss * (self.outside_temp - self.current_temp)

        # 2. Active HVAC influence (scaled by room thermal capacitance)
        if action == ACTION_HEAT:
            delta_hvac = self.heater_power * self.room_scale
        elif action == ACTION_COOL:
            delta_hvac = -self.cooling_power * self.room_scale
        else:
            delta_hvac = 0.0

        # 3. Internal heat gain from occupants (0 if unoccupied)
        effective_occupants = self.num_people if self.people_present else 0
        delta_occupancy = effective_occupants * self.heat_gain_per_person * self.room_scale

        # 4. Random environmental disturbance (e.g. open door, solar spike)
        if disturbance is not None:
            applied_disturbance = float(disturbance)
        elif self.disturbance_level > 0.0:
            applied_disturbance = float(self._rng.normal(0.0, self.disturbance_level))
        else:
            applied_disturbance = 0.0

        # 5. Update temperature
        total_delta = delta_ambient + delta_hvac + delta_occupancy + applied_disturbance
        new_temp = self.current_temp + total_delta

        # Update tracking
        self.current_step += 1
        self.current_temp = new_temp

        # Record step history
        self._record_step(
            step=self.current_step,
            temp=self.current_temp,
            action=action,
            delta_ambient=delta_ambient,
            delta_hvac=delta_hvac,
            delta_occupancy=delta_occupancy,
            disturbance=applied_disturbance,
        )

        return self.get_state()

    def _record_step(
        self,
        step: int,
        temp: float,
        action: str,
        delta_ambient: float,
        delta_hvac: float,
        delta_occupancy: float,
        disturbance: float,
    ) -> None:
        """Internal helper to store snapshot of current state."""
        classification = self.classify_temperature(temp)
        effective_occupants = self.num_people if self.people_present else 0
        self.history.append({
            "step": step,
            "room_temp": round(temp, 2),
            "outside_temp": round(self.outside_temp, 2),
            "desired_temp": round(self.desired_temp, 2),
            "lower_comfort": round(self.desired_temp - self.comfort_tolerance, 2),
            "upper_comfort": round(self.desired_temp + self.comfort_tolerance, 2),
            "action": action,
            "delta_ambient": round(delta_ambient, 3),
            "delta_hvac": round(delta_hvac, 3),
            "delta_occupancy": round(delta_occupancy, 3),
            "disturbance": round(disturbance, 3),
            "people_present": self.people_present,
            "num_people": effective_occupants,
            "room_size": round(self.room_size, 1),
            "classification": classification,
            "is_comfortable": classification == CLASS_COMFORTABLE,
        })

    def get_state(self) -> Dict[str, Any]:
        """Return the current observable state of the environment."""
        effective_occupants = self.num_people if self.people_present else 0
        delta_occ = effective_occupants * self.heat_gain_per_person * self.room_scale
        return {
            "step": self.current_step,
            "room_temp": round(self.current_temp, 2),
            "outside_temp": round(self.outside_temp, 2),
            "desired_temp": round(self.desired_temp, 2),
            "comfort_tolerance": self.comfort_tolerance,
            "classification": self.current_classification,
            "is_comfortable": self.is_comfortable,
            "insulation_level": self.insulation_level,
            "heater_power": self.heater_power,
            "cooling_power": self.cooling_power,
            "disturbance_level": self.disturbance_level,
            "people_present": self.people_present,
            "num_people": effective_occupants,
            "room_size": round(self.room_size, 1),
            "room_scale": round(self.room_scale, 3),
            "heat_gain_per_person": self.heat_gain_per_person,
            "delta_occupancy": round(delta_occ, 3),
        }

    def get_history_df(self) -> pd.DataFrame:
        """Return simulation history as a pandas DataFrame."""
        return pd.DataFrame(self.history)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for the completed or ongoing simulation run."""
        df = self.get_history_df()
        if df.empty or len(df) <= 1:
            return {
                "total_steps": len(df),
                "comfort_percentage": 100.0 if self.is_comfortable else 0.0,
                "avg_error": abs(self.current_temp - self.desired_temp),
                "min_temp": self.current_temp,
                "max_temp": self.current_temp,
            }

        # Exclude step 0 for active metrics if multiple steps exist
        active_df = df.iloc[1:] if len(df) > 1 else df
        comfort_pct = (active_df["is_comfortable"].sum() / len(active_df)) * 100.0
        avg_error = (active_df["room_temp"] - active_df["desired_temp"]).abs().mean()

        return {
            "total_steps": len(active_df),
            "comfort_percentage": round(comfort_pct, 1),
            "avg_error": round(float(avg_error), 2),
            "min_temp": round(float(df["room_temp"].min()), 2),
            "max_temp": round(float(df["room_temp"].max()), 2),
        }

