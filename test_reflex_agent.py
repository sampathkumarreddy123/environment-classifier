"""
Unit and integration test suite verifying the Simple Reflex Thermostat Agent.

Test Conditions (Section 9):
1. Temperature below the lower threshold.
2. Temperature between both thresholds.
3. Temperature above the upper threshold.
4. Temperature fluctuating around a threshold.
5. Different outside temperatures and insulation settings.

Also verifies:
- Threshold validation (lower < upper).
- Pandas DataFrame telemetry schema without utility scores or temperature predictions.
- Mutual exclusion of heating and cooling actions.
- Independent operation alongside the Utility-Based Agent.
"""

import sys
import pandas as pd

from config import ACTION_COOL, ACTION_HEAT, ACTION_OFF
from environment import RoomEnvironment
from reflex_agent import SimpleReflexAgent
from utility_agent import UtilityBasedAgent


def run_all_tests() -> bool:
    print("==================================================")
    print("Running Simple Reflex Thermostat Agent Test Suite")
    print("==================================================")

    agent = SimpleReflexAgent(lower_threshold=20.0, upper_threshold=26.0)

    # -------------------------------------------------------------------------
    # Test 1: Temperature below the lower threshold -> HEAT
    # -------------------------------------------------------------------------
    for temp in [10.0, 18.0, 19.99]:
        dec = agent.decide_action({"room_temp": temp, "step": 1})
        assert dec.selected_action == ACTION_HEAT, (
            f"Expected HEAT at {temp}°C (< 20.0°C), got {dec.selected_action}"
        )
        assert dec.energy_consumed == 1.0
    print("✓ Test 1 Passed: Selects HEAT when temperature is below lower threshold.")

    # -------------------------------------------------------------------------
    # Test 2: Temperature between both thresholds -> OFF
    # -------------------------------------------------------------------------
    for temp in [20.0, 22.0, 23.5, 26.0]:
        dec = agent.decide_action({"room_temp": temp, "step": 2})
        assert dec.selected_action == ACTION_OFF, (
            f"Expected OFF at {temp}°C (within [20.0, 26.0]), got {dec.selected_action}"
        )
        assert dec.energy_consumed == 0.0
    print("✓ Test 2 Passed: Selects OFF when temperature is between thresholds (inclusive).")

    # -------------------------------------------------------------------------
    # Test 3: Temperature above the upper threshold -> COOL
    # -------------------------------------------------------------------------
    for temp in [26.01, 28.5, 35.0]:
        dec = agent.decide_action({"room_temp": temp, "step": 3})
        assert dec.selected_action == ACTION_COOL, (
            f"Expected COOL at {temp}°C (> 26.0°C), got {dec.selected_action}"
        )
        assert dec.energy_consumed == 1.0
    print("✓ Test 3 Passed: Selects COOL when temperature is above upper threshold.")

    # -------------------------------------------------------------------------
    # Test 4: Temperature fluctuating around thresholds
    # -------------------------------------------------------------------------
    agent.reset()
    lower_fluctuations = [
        (19.8, ACTION_HEAT),
        (20.1, ACTION_OFF),
        (19.9, ACTION_HEAT),
        (20.0, ACTION_OFF),
        (19.5, ACTION_HEAT),
    ]
    for step_idx, (temp, expected) in enumerate(lower_fluctuations, start=1):
        dec = agent.decide_action({"room_temp": temp, "step": step_idx})
        assert dec.selected_action == expected, (
            f"Lower threshold oscillation failed at {temp}°C: expected {expected}, got {dec.selected_action}"
        )
        assert dec.selected_action in {ACTION_HEAT, ACTION_COOL, ACTION_OFF}

    upper_fluctuations = [
        (25.9, ACTION_OFF),
        (26.2, ACTION_COOL),
        (26.0, ACTION_OFF),
        (26.1, ACTION_COOL),
        (25.7, ACTION_OFF),
    ]
    for step_idx, (temp, expected) in enumerate(upper_fluctuations, start=6):
        dec = agent.decide_action({"room_temp": temp, "step": step_idx})
        assert dec.selected_action == expected, (
            f"Upper threshold oscillation failed at {temp}°C: expected {expected}, got {dec.selected_action}"
        )
    print("✓ Test 4 Passed: Accurately switches actions when fluctuating around thresholds.")

    # -------------------------------------------------------------------------
    # Test 5: Different outside temperatures and insulation settings
    # -------------------------------------------------------------------------
    # 5a. Single-step decision depends ONLY on current room_temp (ignores outside & insulation)
    for outside in [0.0, 22.0, 42.0]:
        for insulation in ["Poor", "Medium", "Good"]:
            obs_below = {
                "room_temp": 19.0,
                "outside_temp": outside,
                "insulation_level": insulation,
                "desired_temp": 22.0,
            }
            obs_mid = {
                "room_temp": 23.0,
                "outside_temp": outside,
                "insulation_level": insulation,
                "desired_temp": 22.0,
            }
            obs_above = {
                "room_temp": 27.5,
                "outside_temp": outside,
                "insulation_level": insulation,
                "desired_temp": 22.0,
            }
            assert agent.preview_decision(obs_below).selected_action == ACTION_HEAT
            assert agent.preview_decision(obs_mid).selected_action == ACTION_OFF
            assert agent.preview_decision(obs_above).selected_action == ACTION_COOL

    # 5b. Closed-loop multi-step runs across outside temperatures & insulation levels
    scenarios = [
        {"initial_temp": 17.0, "outside_temp": 5.0, "insulation_level": "Poor"},
        {"initial_temp": 22.0, "outside_temp": 38.0, "insulation_level": "Medium"},
        {"initial_temp": 29.0, "outside_temp": 32.0, "insulation_level": "Good"},
    ]
    for scenario in scenarios:
        env = RoomEnvironment(
            initial_temp=scenario["initial_temp"],
            desired_temp=22.0,
            outside_temp=scenario["outside_temp"],
            insulation_level=scenario["insulation_level"],
            heater_power=1.5,
            cooling_power=1.5,
            disturbance_level=0.1,
            random_seed=42,
        )
        loop_agent = SimpleReflexAgent(lower_threshold=20.0, upper_threshold=26.0)
        for _ in range(15):
            rec = loop_agent.step_environment(env)
            # Verify rule consistency on every recorded step
            if rec["room_temp"] < 20.0:
                assert rec["action"] == ACTION_HEAT
            elif rec["room_temp"] > 26.0:
                assert rec["action"] == ACTION_COOL
            else:
                assert rec["action"] == ACTION_OFF

        df = loop_agent.get_history_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 15
        required_cols = {
            "step",
            "room_temp",
            "outside_temp",
            "desired_temp",
            "action",
            "actual_temp",
            "energy_consumed",
            "classification",
        }
        assert required_cols.issubset(set(df.columns)), (
            f"Missing required columns: {required_cols - set(df.columns)}"
        )
        assert "utility_score" not in df.columns
        assert "predicted_temp" not in df.columns

    print("✓ Test 5 Passed: Verified across outside temps, insulation settings, and DataFrame logging.")

    # -------------------------------------------------------------------------
    # Test 6: Threshold validation & dynamic threshold configuration
    # -------------------------------------------------------------------------
    for bad_lower, bad_upper in [(26.0, 20.0), (22.0, 22.0)]:
        raised = False
        try:
            SimpleReflexAgent(lower_threshold=bad_lower, upper_threshold=bad_upper)
        except ValueError:
            raised = True
        assert raised, f"Expected ValueError for lower={bad_lower}, upper={bad_upper}"

    custom_agent = SimpleReflexAgent(lower_threshold=19.0, upper_threshold=24.0)
    custom_agent.update_thresholds(21.0, 25.0)
    rules = custom_agent.get_rules()
    assert rules[0]["label"] == "< 21.0°C → HEAT"
    assert rules[1]["label"] == "21.0°C–25.0°C → OFF"
    assert rules[2]["label"] == "> 25.0°C → COOL"
    assert custom_agent.select_action(20.5) == ACTION_HEAT
    assert custom_agent.select_action(23.0) == ACTION_OFF
    assert custom_agent.select_action(25.5) == ACTION_COOL
    print("✓ Test 6 Passed: Threshold validation and rule configuration verified.")

    # -------------------------------------------------------------------------
    # Test 7: Independent operation alongside Utility-Based Agent
    # -------------------------------------------------------------------------
    env_util = RoomEnvironment(
        initial_temp=18.0,
        desired_temp=22.0,
        outside_temp=34.0,
        insulation_level="Medium",
        disturbance_level=0.1,
        random_seed=42,
    )
    env_refl = RoomEnvironment(
        initial_temp=18.0,
        desired_temp=22.0,
        outside_temp=34.0,
        insulation_level="Medium",
        disturbance_level=0.1,
        random_seed=42,
    )
    util_agent = UtilityBasedAgent()
    refl_agent = SimpleReflexAgent(lower_threshold=20.0, upper_threshold=26.0)

    # Step reflex agent 10 times; utility agent 5 times independently
    for _ in range(10):
        refl_agent.step_environment(env_refl)
    for _ in range(5):
        obs_u = env_util.get_state()
        dec_u = util_agent.decide_action(obs_u)
        env_util.step(dec_u.selected_action)

    assert len(refl_agent.get_history_df()) == 10
    assert env_refl.current_step == 10
    assert len(util_agent.get_decision_history_df()) == 5
    assert env_util.current_step == 5

    # Resetting one agent must not alter the other
    refl_agent.reset()
    env_refl.reset()
    assert len(refl_agent.get_history_df()) == 0
    assert refl_agent.total_energy_consumed == 0.0
    assert len(util_agent.get_decision_history_df()) == 5
    assert env_util.current_step == 5
    print("✓ Test 7 Passed: Simple Reflex and Utility-Based agents operate independently.")

    print("==================================================")
    print("ALL 7 TEST GROUPS PASSED SUCCESSFULLY!")
    print("==================================================")
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

