"""
Verification script for the Room Environment Simulation model and classifier.
"""

import sys
import numpy as np

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
from environment import RoomEnvironment


def run_tests() -> bool:
    print("========================================")
    print("Running Room Environment Foundation Tests")
    print("========================================")

    # 1. Test Initialization
    env = RoomEnvironment(
        initial_temp=18.0,
        desired_temp=22.0,
        outside_temp=30.0,
        insulation_level="Medium",
        heater_power=1.5,
        cooling_power=1.5,
        disturbance_level=0.0,  # deterministic for math test
        comfort_tolerance=1.5,
        random_seed=42,
    )

    assert env.current_step == 0, f"Expected step 0, got {env.current_step}"
    assert env.current_temp == 18.0, f"Expected temp 18.0, got {env.current_temp}"
    assert env.current_classification == CLASS_COLD, f"Expected Cold, got {env.current_classification}"
    print("✓ Test 1: Initialization passed.")

    # 2. Test Classification Logic
    assert env.classify_temperature(20.4) == CLASS_COLD
    assert env.classify_temperature(20.5) == CLASS_COMFORTABLE
    assert env.classify_temperature(22.0) == CLASS_COMFORTABLE
    assert env.classify_temperature(23.5) == CLASS_COMFORTABLE
    assert env.classify_temperature(23.6) == CLASS_HOT
    print("✓ Test 2: Temperature Classification boundaries passed.")

    # 3. Test Thermodynamic Step without HVAC (OFF)
    # delta_ambient = k * (T_out - T_in) = 0.07 * (30.0 - 18.0) = 0.84
    # Expected new temp = 18.0 + 0.84 = 18.84
    state = env.step(action=ACTION_OFF)
    assert env.current_step == 1
    assert np.isclose(env.current_temp, 18.84, atol=1e-3), f"Expected 18.84, got {env.current_temp}"
    print(f"✓ Test 3: Natural heat exchange passed. T_1 = {env.current_temp:.3f}°C")

    # 4. Test Heating Action
    # Previous temp = 18.84
    # delta_ambient = 0.07 * (30.0 - 18.84) = 0.7812
    # delta_hvac = +1.5
    # Expected new temp = 18.84 + 0.7812 + 1.5 = 21.1212
    state = env.step(action=ACTION_HEAT)
    expected_temp = 18.84 + (0.07 * (30.0 - 18.84)) + 1.5
    assert np.isclose(env.current_temp, expected_temp, atol=1e-3), f"Expected {expected_temp:.4f}, got {env.current_temp}"
    assert env.current_classification == CLASS_COMFORTABLE
    print(f"✓ Test 4: Heating step passed. T_2 = {env.current_temp:.3f}°C (Comfortable: {env.is_comfortable})")

    # 5. Test Cooling Action
    prev_temp = env.current_temp
    state = env.step(action=ACTION_COOL)
    expected_ambient = 0.07 * (30.0 - prev_temp)
    expected_cooling_temp = prev_temp + expected_ambient - 1.5
    assert np.isclose(env.current_temp, expected_cooling_temp, atol=1e-3)
    print(f"✓ Test 5: Cooling step passed. T_3 = {env.current_temp:.3f}°C")

    # 6. Test Disturbance Noise
    noisy_env = RoomEnvironment(
        initial_temp=22.0,
        outside_temp=22.0,  # zero ambient delta
        disturbance_level=0.5,
        random_seed=123,
    )
    noisy_env.step(action=ACTION_OFF)
    # delta_ambient = 0, delta_hvac = 0, delta_total = disturbance != 0
    assert not np.isclose(noisy_env.current_temp, 22.0, atol=1e-5)
    print(f"✓ Test 6: Environmental disturbance noise verified. T_1 = {noisy_env.current_temp:.3f}°C")

    # 7. Test History DataFrame & Statistics
    df = env.get_history_df()
    assert len(df) == 4  # step 0, 1, 2, 3
    assert "room_temp" in df.columns
    assert "action" in df.columns
    assert "classification" in df.columns

    stats = env.get_summary_stats()
    assert stats["total_steps"] == 3
    assert "comfort_percentage" in stats
    assert "avg_error" in stats
    print("✓ Test 7: History DataFrame and Summary Statistics verified.")

    # 8. Test Reset
    env.reset(new_initial_temp=25.0)
    assert env.current_step == 0
    assert env.current_temp == 25.0
    assert len(env.get_history_df()) == 1
    print("✓ Test 8: Reset functionality verified.")

    print("========================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("========================================")
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

