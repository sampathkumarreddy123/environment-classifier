"""
Unit tests for Simple Reflex and Utility-Based temperature control agents.
"""

import sys
from config import (
    ACTION_COOL,
    ACTION_HEAT,
    ACTION_OFF,
    CLASS_COLD,
    CLASS_COMFORTABLE,
    CLASS_HOT,
)
from environment import RoomEnvironment
from agents import SimpleReflexAgent, UtilityBasedAgent


def run_agent_tests() -> bool:
    print("========================================")
    print("Running Intelligent Agent Tests")
    print("========================================")

    # 1. Simple Reflex Agent Tests
    reflex = SimpleReflexAgent(hysteresis=0.0)

    # Cold observation
    obs_cold = {
        "room_temp": 18.0,
        "desired_temp": 22.0,
        "outside_temp": 15.0,
        "comfort_tolerance": 1.5,
    }
    action_cold = reflex.act(obs_cold)
    assert action_cold == ACTION_HEAT, f"Expected HEAT, got {action_cold}"
    print(f"✓ Reflex Agent Cold -> {action_cold}")

    # Hot observation
    obs_hot = {
        "room_temp": 26.0,
        "desired_temp": 22.0,
        "outside_temp": 35.0,
        "comfort_tolerance": 1.5,
    }
    action_hot = reflex.act(obs_hot)
    assert action_hot == ACTION_COOL, f"Expected COOL, got {action_hot}"
    print(f"✓ Reflex Agent Hot -> {action_hot}")

    # Comfortable observation
    obs_comfort = {
        "room_temp": 22.2,
        "desired_temp": 22.0,
        "outside_temp": 25.0,
        "comfort_tolerance": 1.5,
    }
    action_comfort = reflex.act(obs_comfort)
    assert action_comfort == ACTION_OFF, f"Expected OFF, got {action_comfort}"
    print(f"✓ Reflex Agent Comfortable -> {action_comfort}")

    # Check metrics
    metrics = reflex.get_metrics()
    assert metrics["total_steps"] == 3
    assert metrics["total_energy"] == 2.0  # 1 HEAT + 1 COOL + 0 OFF
    print("✓ Reflex Agent metrics calculation verified.")

    # 2. Utility-Based Agent Tests
    utility_agent = UtilityBasedAgent(
        weight_comfort=1.0,
        weight_energy=0.4,
        weight_switching=0.1,
    )

    # Very cold room -> Utility should strongly prefer HEAT
    obs_u_cold = {
        "room_temp": 16.0,
        "desired_temp": 22.0,
        "outside_temp": 10.0,
        "comfort_tolerance": 1.5,
        "insulation_level": "Medium",
        "heater_power": 1.5,
        "cooling_power": 1.5,
    }
    action_u_cold = utility_agent.act(obs_u_cold)
    assert action_u_cold == ACTION_HEAT, f"Expected HEAT, got {action_u_cold}"
    assert len(utility_agent.last_action_utilities) == 3
    print(f"✓ Utility Agent Cold -> {action_u_cold} (Utility breakdown verified)")

    # Room already at target comfort -> Energy penalty should make OFF highest utility
    obs_u_comfort = {
        "room_temp": 22.0,
        "desired_temp": 22.0,
        "outside_temp": 22.0,
        "comfort_tolerance": 1.5,
        "insulation_level": "Medium",
        "heater_power": 1.5,
        "cooling_power": 1.5,
    }
    action_u_comfort = utility_agent.act(obs_u_comfort)
    assert action_u_comfort == ACTION_OFF, f"Expected OFF, got {action_u_comfort}"
    print(f"✓ Utility Agent Comfortable -> {action_u_comfort}")

    # 3. Synchronized Dual Simulation Benchmark Test
    env_reflex = RoomEnvironment(
        initial_temp=16.0,
        desired_temp=22.0,
        outside_temp=34.0,
        insulation_level="Medium",
        heater_power=1.5,
        cooling_power=1.5,
        disturbance_level=0.1,
        random_seed=100,
    )

    env_utility = RoomEnvironment(
        initial_temp=16.0,
        desired_temp=22.0,
        outside_temp=34.0,
        insulation_level="Medium",
        heater_power=1.5,
        cooling_power=1.5,
        disturbance_level=0.1,
        random_seed=100,
    )

    agent_r = SimpleReflexAgent()
    agent_u = UtilityBasedAgent()

    for _ in range(25):
        # Reflex step
        obs_r = env_reflex.get_state()
        act_r = agent_r.act(obs_r)
        env_reflex.step(act_r)

        # Utility step
        obs_u = env_utility.get_state()
        act_u = agent_u.act(obs_u)
        env_utility.step(act_u)

    stats_r = env_reflex.get_summary_stats()
    stats_u = env_utility.get_summary_stats()
    metrics_r = agent_r.get_metrics()
    metrics_u = agent_u.get_metrics()

    print(f"✓ Dual Simulation completed 25 steps successfully:")
    print(f"   Reflex: Comfort {stats_r['comfort_percentage']}%, Energy {metrics_r['total_energy']}, Switches {metrics_r['switch_count']}")
    print(f"   Utility: Comfort {stats_u['comfort_percentage']}%, Energy {metrics_u['total_energy']}, Switches {metrics_u['switch_count']}")

    print("========================================")
    print("ALL AGENT TESTS PASSED SUCCESSFULLY!")
    print("========================================")
    return True


if __name__ == "__main__":
    success = run_agent_tests()
    sys.exit(0 if success else 1)

