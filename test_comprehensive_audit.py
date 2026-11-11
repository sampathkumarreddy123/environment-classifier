"""
Comprehensive Master Audit & Verification Test Suite.
Verifies all functional, mathematical, and edge-case requirements:

1. Normal operating conditions
2. Extreme indoor temperatures (10.0°C and 40.0°C)
3. Extreme outdoor temperatures (0.0°C and 45.0°C)
4. Poor and good insulation settings
5. High disturbance levels
6. Minimum and maximum simulation steps (10 and 200 steps)
7. Different utility weights
8. Different reflex thresholds
9. Repeated simulations with identical random seeds (strict reproducibility)
10. Invalid configuration input handling
11. Empty or incomplete simulation data handling
12. Preconfigured demonstration scenarios (Cold, Hot, Comfortable)
"""

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
    INSULATION_FACTORS,
    SimulationDefaults,
)
from environment import RoomEnvironment
from utility_agent import UtilityBasedAgent
from reflex_agent import SimpleReflexAgent
from comparison import (
    ComparisonConfig,
    ComparisonRunner,
    compute_agent_metrics,
    determine_metric_winner,
)


def run_audit() -> bool:
    print("==================================================================")
    print("STARTING COMPREHENSIVE SYSTEM AUDIT & MATHEMATICAL VERIFICATION")
    print("==================================================================")

    # ---------------------------------------------------------------------
    # Test 1: Normal Operating Conditions
    # ---------------------------------------------------------------------
    env = RoomEnvironment(initial_temp=18.0, desired_temp=22.0, outside_temp=30.0, random_seed=42)
    u_agent = UtilityBasedAgent()
    r_agent = SimpleReflexAgent(lower_threshold=20.5, upper_threshold=23.5)

    u_dec = u_agent.decide_action(env.get_state())
    r_dec = r_agent.decide_action(env.get_state())
    assert u_dec.selected_action in [ACTION_HEAT, ACTION_COOL, ACTION_OFF]
    assert r_dec.selected_action == ACTION_HEAT  # 18.0 < 20.5
    print("✓ Test 1: Normal operating conditions verified.")

    # ---------------------------------------------------------------------
    # Test 2: Extreme Indoor Temperatures (10.0°C and 40.0°C)
    # ---------------------------------------------------------------------
    # 2a. Extreme Cold Indoor (10.0°C, target 22.0°C)
    env_cold = RoomEnvironment(initial_temp=10.0, desired_temp=22.0, outside_temp=15.0)
    obs_cold = env_cold.get_state()
    assert env_cold.current_classification == CLASS_COLD
    assert u_agent.decide_action(obs_cold).selected_action == ACTION_HEAT
    assert r_agent.decide_action(obs_cold).selected_action == ACTION_HEAT

    # 2b. Extreme Hot Indoor (40.0°C, target 22.0°C)
    env_hot = RoomEnvironment(initial_temp=40.0, desired_temp=22.0, outside_temp=35.0)
    obs_hot = env_hot.get_state()
    assert env_hot.current_classification == CLASS_HOT
    assert u_agent.decide_action(obs_hot).selected_action == ACTION_COOL
    assert r_agent.decide_action(obs_hot).selected_action == ACTION_COOL
    print("✓ Test 2: Extreme indoor temperatures (10°C & 40°C) handled correctly.")

    # ---------------------------------------------------------------------
    # Test 3: Extreme Outdoor Temperatures (0.0°C and 45.0°C)
    # ---------------------------------------------------------------------
    # 3a. Freezing Outdoor (0.0°C) with comfortable room (22.0°C)
    env_freezing = RoomEnvironment(initial_temp=22.0, desired_temp=22.0, outside_temp=0.0)
    obs_freezing = env_freezing.get_state()
    u_freeze_dec = u_agent.decide_action(obs_freezing)
    # Due to severe outdoor cold (delta ambient = 0.07 * (0 - 22) = -1.54°C),
    # Utility agent anticipates rapid heat loss and triggers HEAT to maintain comfort!
    assert u_freeze_dec.selected_action == ACTION_HEAT

    # 3b. Extreme Heatwave Outdoor (45.0°C) with comfortable room (22.0°C)
    env_heatwave = RoomEnvironment(initial_temp=22.0, desired_temp=22.0, outside_temp=45.0)
    obs_heatwave = env_heatwave.get_state()
    u_heat_dec = u_agent.decide_action(obs_heatwave)
    assert u_heat_dec.selected_action == ACTION_COOL
    print("✓ Test 3: Extreme outdoor temperatures (0°C & 45°C) and lookahead compensation verified.")

    # ---------------------------------------------------------------------
    # Test 4: Poor vs. Good Insulation Dynamics
    # ---------------------------------------------------------------------
    env_poor = RoomEnvironment(initial_temp=20.0, outside_temp=35.0, insulation_level="Poor")
    env_good = RoomEnvironment(initial_temp=20.0, outside_temp=35.0, insulation_level="Good")
    assert env_poor.k_loss == INSULATION_FACTORS["Poor"]  # 0.14
    assert env_good.k_loss == INSULATION_FACTORS["Good"]  # 0.025
    
    # Passive step (OFF)
    state_poor = env_poor.step(ACTION_OFF, disturbance=0.0)
    state_good = env_good.step(ACTION_OFF, disturbance=0.0)
    # Poor insulation should allow substantially more heat transfer:
    # delta_poor = 0.14 * (35 - 20) = 2.1°C -> 22.1°C
    # delta_good = 0.025 * (35 - 20) = 0.375°C -> 20.375°C
    assert state_poor["room_temp"] > state_good["room_temp"]
    assert np.isclose(state_poor["room_temp"], 22.1, atol=1e-2)
    assert np.isclose(state_good["room_temp"], 20.375, atol=1e-2)
    print("✓ Test 4: Poor vs. Good insulation heat transfer physics verified.")

    # ---------------------------------------------------------------------
    # Test 5: High Disturbance Levels (Robustness & Clamping)
    # ---------------------------------------------------------------------
    env_dist = RoomEnvironment(
        initial_temp=22.0,
        disturbance_level=2.0,
        random_seed=123,
    )
    for _ in range(10):
        s = env_dist.step(ACTION_OFF)
        assert -50.0 < s["room_temp"] < 100.0
    print("✓ Test 5: High disturbance level (sigma = 2.0°C) simulated stably.")

    # ---------------------------------------------------------------------
    # Test 6: Minimum (10) and Maximum (200) Simulation Steps
    # ---------------------------------------------------------------------
    cfg_min = ComparisonConfig(max_steps=10, random_seed=99)
    runner_min = ComparisonRunner(cfg_min)
    runner_min.run_all()
    res_min = runner_min.evaluate()
    assert len(res_min["combined_df"]) == 11  # Step 0 + 10 steps

    cfg_max = ComparisonConfig(max_steps=200, random_seed=99)
    runner_max = ComparisonRunner(cfg_max)
    runner_max.run_all()
    res_max = runner_max.evaluate()
    assert len(res_max["combined_df"]) == 201  # Step 0 + 200 steps
    print("✓ Test 6: Simulation horizon limits (10 to 200 steps) verified.")

    # ---------------------------------------------------------------------
    # Test 7: Different Utility Weights Tuning
    # ---------------------------------------------------------------------
    # Agent with extreme comfort priority
    agent_comfort = UtilityBasedAgent(weight_comfort=3.0, weight_energy=0.0)
    # Agent with extreme energy saving priority
    agent_energy = UtilityBasedAgent(weight_comfort=0.1, weight_energy=3.0)

    # Slight deviation: Room at 20.0°C, target 22.0°C, tolerance 1.5°C (lower comfort 20.5°C)
    test_obs = {
        "room_temp": 20.2,
        "desired_temp": 22.0,
        "outside_temp": 21.0,
        "insulation_level": "Medium",
        "heater_power": 1.5,
        "cooling_power": 1.5,
        "comfort_tolerance": 1.5,
        "step": 1,
    }
    dec_c = agent_comfort.decide_action(test_obs)
    dec_e = agent_energy.decide_action(test_obs)
    assert dec_c.selected_action == ACTION_HEAT  # Comfort prioritized
    assert dec_e.selected_action == ACTION_OFF   # Energy conservation prioritized
    print("✓ Test 7: Utility weights tuning impact verified.")

    # ---------------------------------------------------------------------
    # Test 8: Different Reflex Thresholds
    # ---------------------------------------------------------------------
    # Narrow band [21.5, 22.5] vs Wide band [18.0, 26.0]
    r_narrow = SimpleReflexAgent(lower_threshold=21.5, upper_threshold=22.5)
    r_wide = SimpleReflexAgent(lower_threshold=18.0, upper_threshold=26.0)
    
    assert r_narrow.select_action(21.0) == ACTION_HEAT
    assert r_wide.select_action(21.0) == ACTION_OFF  # Inside [18.0, 26.0]
    print("✓ Test 8: Reflex threshold variations verified.")

    # ---------------------------------------------------------------------
    # Test 9: Strict Re-run Determinism with Identical Random Seeds
    # ---------------------------------------------------------------------
    cfg_det = ComparisonConfig(random_seed=777, max_steps=30)
    runner_1 = ComparisonRunner(cfg_det)
    runner_2 = ComparisonRunner(cfg_det)
    runner_1.run_all()
    runner_2.run_all()
    res_1 = runner_1.evaluate()
    res_2 = runner_2.evaluate()

    pd.testing.assert_frame_equal(res_1["utility_df"], res_2["utility_df"])
    pd.testing.assert_frame_equal(res_1["reflex_df"], res_2["reflex_df"])
    pd.testing.assert_frame_equal(res_1["combined_df"], res_2["combined_df"])
    print("✓ Test 9: Exact same-seed determinism verified bit-for-bit across runs.")

    # ---------------------------------------------------------------------
    # Test 10: Invalid Configuration Input Handling
    # ---------------------------------------------------------------------
    try:
        # Invalid thresholds where lower >= upper
        SimpleReflexAgent(lower_threshold=25.0, upper_threshold=20.0)
        assert False, "Expected ValueError when lower_threshold >= upper_threshold"
    except ValueError:
        pass

    try:
        r_agent.update_thresholds(24.0, 24.0)
        assert False, "Expected ValueError when updating lower == upper"
    except ValueError:
        pass
    print("✓ Test 10: Invalid configuration input detection and exceptions verified.")

    # ---------------------------------------------------------------------
    # Test 11: Empty / Incomplete Simulation Data Handlers
    # ---------------------------------------------------------------------
    empty_df = pd.DataFrame()
    metrics_empty = compute_agent_metrics(empty_df)
    assert metrics_empty.total_steps == 0
    assert metrics_empty.avg_abs_error == 0.0
    assert metrics_empty.total_energy == 0.0

    step0_df = pd.DataFrame([{
        "step": 0,
        "room_temp": 20.0,
        "outside_temp": 25.0,
        "desired_temp": 22.0,
        "lower_comfort": 20.5,
        "upper_comfort": 23.5,
        "action": ACTION_OFF,
        "actual_temp": 20.0,
        "energy_consumed": 0.0,
        "classification": CLASS_COLD,
    }])
    metrics_step0 = compute_agent_metrics(step0_df)
    assert metrics_step0.total_steps == 0
    assert metrics_step0.total_energy == 0.0
    print("✓ Test 11: Empty and step-0 simulation telemetry handled safely without NaNs or crashes.")

    # ---------------------------------------------------------------------
    # Test 12: Preconfigured Demonstration Scenarios
    # ---------------------------------------------------------------------
    for sc_id, sc in DEMO_SCENARIOS.items():
        cfg_sc = ComparisonConfig(
            initial_temp=sc.initial_temp,
            desired_temp=sc.desired_temp,
            outside_temp=sc.outside_temp,
            insulation_level=sc.insulation_level,
            comfort_tolerance=sc.comfort_tolerance,
            heater_power=sc.heater_power,
            cooling_power=sc.cooling_power,
            disturbance_level=sc.disturbance_level,
            max_steps=sc.max_steps,
            random_seed=sc.random_seed,
            lower_threshold=sc.lower_threshold,
            upper_threshold=sc.upper_threshold,
            weight_comfort=sc.weight_comfort,
            weight_energy=sc.weight_energy,
            weight_stability=sc.weight_stability,
            weight_overshoot=sc.weight_overshoot,
        )
        sc_runner = ComparisonRunner(cfg_sc)
        sc_runner.run_all()
        sc_res = sc_runner.evaluate()
        assert len(sc_res["combined_df"]) == sc.max_steps + 1
        assert sc_res["utility_metrics"].total_steps == sc.max_steps
        assert sc_res["reflex_metrics"].total_steps == sc.max_steps
        print(f"  ✓ {sc.name}: Simulated {sc.max_steps} steps successfully.")
    print("✓ Test 12: All 3 preconfigured academic demonstration scenarios verified.")

    print("==================================================================")
    print("ALL 12 COMPREHENSIVE AUDIT & VERIFICATION TESTS PASSED (100% OK)")
    print("==================================================================")
    return True


if __name__ == "__main__":
    ok = run_audit()
    sys.exit(0 if ok else 1)
