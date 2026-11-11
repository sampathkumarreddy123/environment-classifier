"""
Unit test suite verifying the Utility-Based Intelligent Agent across all required conditions.

Test Conditions:
1. Room temperature significantly below target (Cold start).
2. Room temperature significantly above target (Hot start).
3. Room temperature already comfortable.
4. Very cold outside temperature (Lookahead proactive heating).
5. Poor room insulation (High heat dissipation response).
"""

import sys
import numpy as np
import pandas as pd

from config import ACTION_COOL, ACTION_HEAT, ACTION_OFF, ALL_ACTIONS
from environment import RoomEnvironment
from utility_agent import UtilityBasedAgent


def run_all_tests() -> bool:
    print("==================================================")
    print("Running Utility-Based Intelligent Agent Test Suite")
    print("==================================================")

    agent = UtilityBasedAgent(
        weight_comfort=1.0,
        weight_energy=0.5,
        weight_stability=0.2,
        weight_overshoot=1.0,
    )

    # -------------------------------------------------------------------------
    # Test 1: Room temperature significantly below target
    # -------------------------------------------------------------------------
    obs_cold = {
        "room_temp": 12.0,
        "desired_temp": 24.0,
        "outside_temp": 15.0,
        "insulation_level": "Medium",
        "heater_power": 1.5,
        "cooling_power": 1.5,
        "comfort_tolerance": 1.5,
        "step": 1,
    }
    decision_cold = agent.decide_action(obs_cold)
    scores_cold = agent.get_action_scores_dict()

    print(f"Test 1 - Cold Room (12°C -> Target 24°C):")
    print(f"  Selected: {decision_cold.selected_action} (Utility: {decision_cold.utility_score})")
    print(f"  All Scores: {scores_cold}")
    print(f"  Reason: {decision_cold.reason}")

    assert decision_cold.selected_action == ACTION_HEAT, f"Expected HEAT, got {decision_cold.selected_action}"
    assert scores_cold[ACTION_HEAT] > scores_cold[ACTION_OFF], "HEAT utility must exceed OFF utility"
    assert scores_cold[ACTION_HEAT] > scores_cold[ACTION_COOL], "HEAT utility must exceed COOL utility"
    assert len(decision_cold.action_evaluations) == 3, "Must evaluate all three actions"
    print("✓ Test 1 Passed: Agent selects HEAT when significantly below target.\n")

    # -------------------------------------------------------------------------
    # Test 2: Room temperature significantly above target
    # -------------------------------------------------------------------------
    obs_hot = {
        "room_temp": 32.0,
        "desired_temp": 22.0,
        "outside_temp": 35.0,
        "insulation_level": "Medium",
        "heater_power": 1.5,
        "cooling_power": 1.5,
        "comfort_tolerance": 1.5,
        "step": 2,
    }
    decision_hot = agent.decide_action(obs_hot)
    scores_hot = agent.get_action_scores_dict()

    print(f"Test 2 - Hot Room (32°C -> Target 22°C):")
    print(f"  Selected: {decision_hot.selected_action} (Utility: {decision_hot.utility_score})")
    print(f"  All Scores: {scores_hot}")

    assert decision_hot.selected_action == ACTION_COOL, f"Expected COOL, got {decision_hot.selected_action}"
    assert scores_hot[ACTION_COOL] > scores_hot[ACTION_OFF], "COOL utility must exceed OFF utility"
    assert scores_hot[ACTION_COOL] > scores_hot[ACTION_HEAT], "COOL utility must exceed HEAT utility"
    print("✓ Test 2 Passed: Agent selects COOL when significantly above target.\n")

    # -------------------------------------------------------------------------
    # Test 3: Room temperature already comfortable
    # -------------------------------------------------------------------------
    obs_comfy = {
        "room_temp": 22.0,
        "desired_temp": 22.0,
        "outside_temp": 22.0,
        "insulation_level": "Medium",
        "heater_power": 1.5,
        "cooling_power": 1.5,
        "comfort_tolerance": 1.5,
        "step": 3,
    }
    decision_comfy = agent.decide_action(obs_comfy)
    scores_comfy = agent.get_action_scores_dict()

    print(f"Test 3 - Comfortable Room (22°C -> Target 22°C, Outside 22°C):")
    print(f"  Selected: {decision_comfy.selected_action} (Utility: {decision_comfy.utility_score})")
    print(f"  All Scores: {scores_comfy}")

    assert decision_comfy.selected_action == ACTION_OFF, f"Expected OFF, got {decision_comfy.selected_action}"
    assert scores_comfy[ACTION_OFF] > scores_comfy[ACTION_HEAT], "OFF must be superior to unnecessary HEAT"
    assert scores_comfy[ACTION_OFF] > scores_comfy[ACTION_COOL], "OFF must be superior to unnecessary COOL"
    print("✓ Test 3 Passed: Agent selects OFF to conserve energy when comfortable.\n")

    # -------------------------------------------------------------------------
    # Test 4: Very cold outside temperature (proactive thermal compensation)
    # -------------------------------------------------------------------------
    # Room is currently at 21.0°C (inside comfort band [20.5, 23.5]),
    # but outside is 0°C. If OFF is chosen, ambient loss = 0.07 * (0 - 21) = -1.47°C,
    # causing temperature to drop to 19.53°C (cold violation and high comfort penalty!).
    # If HEAT is chosen, heating balances loss (+1.5 - 1.47 = +0.03°C), maintaining 21.03°C.
    obs_freezing_out = {
        "room_temp": 21.0,
        "desired_temp": 22.0,
        "outside_temp": 0.0,
        "insulation_level": "Medium",
        "heater_power": 1.5,
        "cooling_power": 1.5,
        "comfort_tolerance": 1.5,
        "step": 4,
    }
    decision_freezing = agent.decide_action(obs_freezing_out)
    scores_freezing = agent.get_action_scores_dict()

    print(f"Test 4 - Freezing Outside (Outside 0°C, Room 21°C -> Target 22°C):")
    print(f"  Predicted if OFF: {decision_freezing.action_evaluations[ACTION_OFF].predicted_temp}°C")
    print(f"  Predicted if HEAT: {decision_freezing.action_evaluations[ACTION_HEAT].predicted_temp}°C")
    print(f"  Selected: {decision_freezing.selected_action}")
    print(f"  All Scores: {scores_freezing}")

    assert decision_freezing.selected_action == ACTION_HEAT, f"Expected proactive HEAT, got {decision_freezing.selected_action}"
    assert scores_freezing[ACTION_HEAT] > scores_freezing[ACTION_OFF], "HEAT must beat OFF due to ambient heat loss"
    print("✓ Test 4 Passed: Agent proactively heats against harsh outdoor freeze.\n")

    # -------------------------------------------------------------------------
    # Test 5: Poor room insulation
    # -------------------------------------------------------------------------
    # Under poor insulation (k = 0.14), heat leaks twice as fast.
    obs_poor_ins = {
        "room_temp": 18.0,
        "desired_temp": 22.0,
        "outside_temp": 8.0,
        "insulation_level": "Poor",
        "heater_power": 1.5,
        "cooling_power": 1.5,
        "comfort_tolerance": 1.5,
        "step": 5,
    }
    decision_poor = agent.decide_action(obs_poor_ins)
    scores_poor = agent.get_action_scores_dict()

    print(f"Test 5 - Poor Insulation (k=0.14, Room 18°C, Outside 8°C -> Target 22°C):")
    print(f"  Predicted if HEAT: {decision_poor.action_evaluations[ACTION_HEAT].predicted_temp}°C")
    print(f"  Predicted if OFF: {decision_poor.action_evaluations[ACTION_OFF].predicted_temp}°C")
    print(f"  Selected: {decision_poor.selected_action}")
    print(f"  All Scores: {scores_poor}")

    assert decision_poor.selected_action == ACTION_HEAT
    assert scores_poor[ACTION_HEAT] > scores_poor[ACTION_OFF]
    print("✓ Test 5 Passed: Agent accounts for high heat loss under poor insulation.\n")

    # -------------------------------------------------------------------------
    # Test 6: Closed-loop multi-step simulation with RoomEnvironment
    # -------------------------------------------------------------------------
    print("Test 6 - Multi-step Closed Loop Simulation:")
    env = RoomEnvironment(
        initial_temp=14.0,
        desired_temp=22.0,
        outside_temp=30.0,
        insulation_level="Medium",
        heater_power=1.5,
        cooling_power=1.5,
        disturbance_level=0.1,
        random_seed=42,
    )
    test_agent = UtilityBasedAgent()

    for step in range(15):
        obs = env.get_state()
        dec = test_agent.decide_action(obs)
        env.step(dec.selected_action)

    df_decisions = test_agent.get_decision_history_df()
    assert len(df_decisions) == 15, f"Expected 15 records, got {len(df_decisions)}"
    assert "utility_score" in df_decisions.columns
    assert "predicted_temp" in df_decisions.columns
    assert "action" in df_decisions.columns

    print(f"  Simulation completed 15 steps. Final temp: {env.current_temp:.2f}°C")
    print(f"  Classification: {env.current_classification}")
    print("✓ Test 6 Passed: Multi-step simulation and dataframe logging validated.\n")

    print("==================================================")
    print("ALL 6 TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

