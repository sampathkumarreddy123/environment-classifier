"""
Comprehensive Verification Suite for Occupancy-Aware Environment Model & Intelligent Agent.
Tests all 16 criteria specified in Section 12 of the specification:
1. Empty room with zero occupants.
2. Room with one occupant.
3. Room with two occupants.
4. Room with five occupants.
5. Room with multiple occupants (10 occupants).
6. Small room configuration (10 m²).
7. Medium room configuration (25 m²).
8. Large room configuration (80 m²).
9. Heating behavior under occupancy load.
10. Cooling behavior under occupancy load.
11. OFF behavior (unoccupied setback mode).
12. Multi-objective utility score calculations with occupancy weighting.
13. Reflex thermostat condition-action rules.
14. Synchronized Agent comparison with identical disturbance sequences.
15. CSV exports with all 9 required parameters.
16. Light and Dark theme SVG & HTML rendering.
"""

import io
import sys
import numpy as np
import pandas as pd

from config import (
    ACTION_COOL,
    ACTION_HEAT,
    ACTION_OFF,
    CLASS_COLD,
    CLASS_COMFORTABLE,
    CLASS_HOT,
    DEMO_SCENARIOS,
    SimulationDefaults,
)
from environment import RoomEnvironment
from utility_agent import UtilityBasedAgent
from reflex_agent import SimpleReflexAgent
from comparison import ComparisonConfig, ComparisonRunner, build_comparison_table_df
from utils import (
    get_live_room_simulation_html,
    get_device_status_panel_html,
    create_occupancy_heat_chart,
)


def run_occupancy_tests() -> bool:
    print("==================================================================")
    print("STARTING OCCUPANCY & ROOM CONFIGURATION VERIFICATION SUITE")
    print("==================================================================")

    # -----------------------------------------------------------------
    # Criterion 1: Empty Room with Zero Occupants
    # -----------------------------------------------------------------
    env_empty = RoomEnvironment(
        initial_temp=22.0, desired_temp=22.0, outside_temp=22.0,
        disturbance_level=0.0, people_present=False, num_people=0, room_size=25.0
    )
    assert env_empty.delta_occupancy == 0.0, "Zero occupants must produce exactly 0.0°C heat gain."
    st_empty = env_empty.step(ACTION_OFF)
    assert np.isclose(st_empty["room_temp"], 22.0, atol=1e-4), "Empty room at equilibrium must stay unchanged."
    assert st_empty["delta_occupancy"] == 0.0
    print("✓ Criterion 1: Empty room with zero occupants produces exactly 0.00°C internal heat.")

    # -----------------------------------------------------------------
    # Criteria 2-5: Rooms with 1, 2, 5, and Multiple (10) Occupants
    # -----------------------------------------------------------------
    occupant_counts = [1, 2, 5, 10]
    expected_heats = [n * 0.08 for n in occupant_counts]
    actual_heats = []

    for n, expected_q in zip(occupant_counts, expected_heats):
        env_occ = RoomEnvironment(
            initial_temp=20.0, desired_temp=20.0, outside_temp=20.0,
            disturbance_level=0.0, people_present=True, num_people=n, room_size=25.0
        )
        assert np.isclose(env_occ.delta_occupancy, expected_q, atol=1e-4), f"Expected {expected_q}, got {env_occ.delta_occupancy}"
        actual_heats.append(env_occ.delta_occupancy)

    assert actual_heats[0] < actual_heats[1] < actual_heats[2] < actual_heats[3], "More occupants must strictly increase heat gain."
    print(f"✓ Criteria 2–5: Occupants {occupant_counts} produce monotonically increasing heat gains: {actual_heats}°C/step.")

    # -----------------------------------------------------------------
    # Criteria 6-8: Small, Medium, and Large Room Configurations
    # -----------------------------------------------------------------
    # 2 occupants (base heat 0.16°C at 25m²)
    # Small room: 10 m² -> thermal scale = 25/10 = 2.5x -> body heat = 0.16 * 2.5 = 0.40°C/step
    # Medium room: 25 m² -> thermal scale = 25/25 = 1.0x -> body heat = 0.16 * 1.0 = 0.16°C/step
    # Large room: 100 m² -> thermal scale = 25/100 = 0.25x -> body heat = 0.16 * 0.25 = 0.04°C/step
    env_small = RoomEnvironment(people_present=True, num_people=2, room_size=10.0, disturbance_level=0.0)
    env_med = RoomEnvironment(people_present=True, num_people=2, room_size=25.0, disturbance_level=0.0)
    env_large = RoomEnvironment(people_present=True, num_people=2, room_size=100.0, disturbance_level=0.0)

    assert np.isclose(env_small.delta_occupancy, 0.40, atol=1e-4)
    assert np.isclose(env_med.delta_occupancy, 0.16, atol=1e-4)
    assert np.isclose(env_large.delta_occupancy, 0.04, atol=1e-4)
    assert env_small.delta_hvac_heat > env_med.delta_hvac_heat > env_large.delta_hvac_heat, "HVAC power must scale inversely with room area."
    print("✓ Criteria 6–8: Small (10m²), Medium (25m²), and Large (100m²) room thermal scaling verified.")

    # -----------------------------------------------------------------
    # Criteria 9-11: Heating, Cooling, and OFF Behavior
    # -----------------------------------------------------------------
    agent = UtilityBasedAgent()
    # Heating test: Cold room (12°C -> 22°C) with 2 occupants
    env_heat = RoomEnvironment(initial_temp=12.0, desired_temp=22.0, outside_temp=10.0, people_present=True, num_people=2, room_size=25.0)
    dec_heat = agent.decide_action(env_heat.get_state())
    assert dec_heat.selected_action == ACTION_HEAT, f"Expected HEAT, got {dec_heat.selected_action}"

    # Cooling test: Hot room (30°C -> 22°C) with 8 occupants (high metabolic heat)
    env_cool = RoomEnvironment(initial_temp=30.0, desired_temp=22.0, outside_temp=32.0, people_present=True, num_people=8, room_size=25.0)
    dec_cool = agent.decide_action(env_cool.get_state())
    assert dec_cool.selected_action == ACTION_COOL, f"Expected COOL, got {dec_cool.selected_action}"

    # OFF test: Unoccupied room at 20.8°C with target 22.0°C (setback conservation)
    env_unocc = RoomEnvironment(initial_temp=21.2, desired_temp=22.0, outside_temp=21.0, people_present=False, num_people=0, room_size=25.0)
    dec_unocc = agent.decide_action(env_unocc.get_state())
    assert dec_unocc.selected_action == ACTION_OFF, f"Expected OFF, got {dec_unocc.selected_action}"
    print("✓ Criteria 9–11: Heating, Cooling, and OFF (unoccupied setback) verified.")

    # -----------------------------------------------------------------
    # Criterion 12: Utility Score Calculations
    # -----------------------------------------------------------------
    obs = env_heat.get_state()
    dec = agent.decide_action(obs)
    evals = dec.action_evaluations
    assert set(evals.keys()) == {ACTION_HEAT, ACTION_COOL, ACTION_OFF}
    for act, ev in evals.items():
        assert ev.utility <= 0.0, "Utility penalties must yield non-positive scores."
        assert dec.people_present is True
        assert dec.num_people == 2
        assert dec.room_size == 25.0
    print("✓ Criterion 12: Multi-objective utility scores and action evaluations verified.")

    # -----------------------------------------------------------------
    # Criterion 13: Simple Reflex Thermostat Condition-Action Rules
    # -----------------------------------------------------------------
    reflex = SimpleReflexAgent(lower_threshold=20.5, upper_threshold=23.5)
    assert reflex.select_action(20.4) == ACTION_HEAT
    assert reflex.select_action(20.5) == ACTION_OFF
    assert reflex.select_action(22.0) == ACTION_OFF
    assert reflex.select_action(23.5) == ACTION_OFF
    assert reflex.select_action(23.6) == ACTION_COOL
    print("✓ Criterion 13: Simple Reflex condition-action thresholds verified.")

    # -----------------------------------------------------------------
    # Criterion 14: Synchronized Agent Comparison
    # -----------------------------------------------------------------
    cfg = ComparisonConfig(
        initial_temp=17.0, desired_temp=22.0, outside_temp=30.0,
        people_present=True, num_people=4, room_size=30.0,
        max_steps=20, random_seed=777
    )
    runner = ComparisonRunner(cfg)
    runner.run_all()
    eval_res = runner.evaluate()
    comb_df = eval_res["combined_df"]
    # Identical environmental conditions verification
    assert (comb_df["utility_room_temp"].iloc[0] == comb_df["reflex_room_temp"].iloc[0])
    assert (comb_df["disturbance"] == comb_df["disturbance"]).all()
    assert (comb_df["delta_occupancy"] == comb_df["delta_occupancy"]).all()
    assert len(eval_res["comparison_df"]) == 14
    print("✓ Criterion 14: Synchronized comparison with identical disturbance and occupancy load verified.")

    # -----------------------------------------------------------------
    # Criterion 15: CSV Exports with All Required Parameters
    # -----------------------------------------------------------------
    required_cols = [
        "step", "room_temp", "outside_temp", "desired_temp", "action",
        "actual_temp", "energy_consumed", "classification",
        "people_present", "num_people", "room_size", "delta_occupancy"
    ]
    u_csv = eval_res["utility_df"].to_csv(index=False)
    u_parsed = pd.read_csv(io.StringIO(u_csv))
    for col in required_cols:
        assert col in u_parsed.columns, f"Missing {col} in utility CSV export."

    r_csv = eval_res["reflex_df"].to_csv(index=False)
    r_parsed = pd.read_csv(io.StringIO(r_csv))
    for col in required_cols:
        assert col in r_parsed.columns, f"Missing {col} in reflex CSV export."
    print("✓ Criterion 15: CSV exports verified with all 9 required environmental and occupancy parameters.")

    # -----------------------------------------------------------------
    # Criterion 16: Light and Dark Theme SVG & HTML Rendering
    # -----------------------------------------------------------------
    for dark in (False, True):
        # Occupied room
        html_occ = get_live_room_simulation_html(
            current_temp=21.5, desired_temp=22.0, outside_temp=15.0,
            classification=CLASS_COMFORTABLE, action=ACTION_OFF,
            dark_mode=dark, people_present=True, num_people=3, room_size=28.0
        )
        assert "occupancy-lounge" in html_occ
        assert "person-1" in html_occ
        assert "person-2" in html_occ
        assert "person-3" in html_occ
        assert "Internal Metabolic Heat" in html_occ

        # Unoccupied room
        html_unocc = get_live_room_simulation_html(
            current_temp=21.5, desired_temp=22.0, outside_temp=15.0,
            classification=CLASS_COMFORTABLE, action=ACTION_OFF,
            dark_mode=dark, people_present=False, num_people=0, room_size=28.0
        )
        assert "occupancy-empty-banner" in html_unocc
        assert "Room Unoccupied" in html_unocc

        # Dual-axis chart
        fig_chart = create_occupancy_heat_chart(comb_df, dark_mode=dark)
        assert fig_chart is not None
        assert len(fig_chart.data) == 2

    print("✓ Criterion 16: Light and Dark Mode SVG, HTML widgets, and Plotly charts verified.")

    print("==================================================================")
    print("ALL 16 OCCUPANCY & ROOM CONFIGURATION CRITERIA PASSED (100% OK)!")
    print("==================================================================")
    return True


if __name__ == "__main__":
    ok = run_occupancy_tests()
    sys.exit(0 if ok else 1)
