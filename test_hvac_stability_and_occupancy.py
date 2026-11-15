"""
Comprehensive Test Suite for HVAC Mode Stability (Anti-Chattering)
and Occupancy-Aware Utility Decision-Making (Section 10 Verification).
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
    HVACControlDefaults,
)
from environment import RoomEnvironment
from reflex_agent import SimpleReflexAgent
from utility_agent import UtilityBasedAgent
from comparison import ComparisonConfig, ComparisonRunner


def run_all_verification_tests() -> bool:
    print("==================================================================")
    print("STARTING HVAC STABILITY & OCCUPANCY-AWARE VERIFICATION SUITE")
    print("==================================================================")

    # =========================================================================
    # PART 1: HVAC STABILITY & ANTI-CHATTERING TESTS
    # =========================================================================
    print("\n--- PART 1: HVAC STABILITY & ANTI-CHATTERING ---")

    # 1.1 Temperature near target (20.7°C with target 20.0°C)
    # Verify no 1-step toggling or hunting
    r_agent = SimpleReflexAgent(
        lower_threshold=19.5,
        upper_threshold=20.5,
        upper_heating_threshold=20.5,
        lower_cooling_threshold=19.5,
        min_on_duration=2,
        min_off_duration=2,
    )
    # Cooling activates above 20.5°C
    dec1 = r_agent.decide_action({"room_temp": 20.7, "step": 1})
    assert dec1.selected_action == ACTION_COOL, f"Expected COOL at 20.7°C, got {dec1.selected_action}"
    
    # Next step, temp drops to 20.1°C: min duration holds COOL (prevent rapid toggle to OFF)
    dec2 = r_agent.decide_action({"room_temp": 20.1, "step": 2})
    assert dec2.selected_action == ACTION_COOL, f"Expected COOL maintained at 20.1°C (min duration), got {dec2.selected_action}"
    
    # Step 3, temp reaches lower cooling threshold (19.5°C): deactivation to OFF
    dec3 = r_agent.decide_action({"room_temp": 19.5, "step": 3})
    assert dec3.selected_action == ACTION_OFF, f"Expected OFF when lower cutoff reached, got {dec3.selected_action}"

    # Step 4, inside deadband at 19.8°C: must remain OFF
    dec4 = r_agent.decide_action({"room_temp": 19.8, "step": 4})
    assert dec4.selected_action == ACTION_OFF, f"Expected OFF inside deadband, got {dec4.selected_action}"
    print("✓ 1.1: Reflex hysteresis deadband & minimum operating duration verified (no toggling at 20.7°C).")

    # 1.2 Utility Agent near target anti-chattering test
    u_agent = UtilityBasedAgent()
    # At 20.7°C (target 20.0°C, tolerance 1.5°C): inside comfort deadband [18.5, 21.5]
    obs_near = {
        "room_temp": 20.7,
        "desired_temp": 20.0,
        "outside_temp": 25.0,
        "comfort_tolerance": 1.5,
        "people_present": True,
        "num_people": 2,
        "step": 1,
    }
    dec_u1 = u_agent.decide_action(obs_near)
    # Inside comfort zone with mild outside temp, passive drift maintains comfort: OFF is optimal
    assert dec_u1.selected_action == ACTION_OFF
    print("✓ 1.2: Utility Agent avoids unnecessary conditioning inside deadband.")

    # 1.3 Anti-Reversal: No immediate flip-flop from HEAT to COOL
    u_agent.reset()
    # Force agent into HEAT
    u_agent.decide_action({
        "room_temp": 15.0, "desired_temp": 22.0, "outside_temp": 15.0,
        "people_present": True, "num_people": 2, "step": 1
    })
    assert u_agent.last_action == ACTION_HEAT

    # Next step: temperature reaches 21.8°C (near target 22°C)
    # Must NOT directly reverse to COOL!
    dec_next = u_agent.decide_action({
        "room_temp": 21.8, "desired_temp": 22.0, "outside_temp": 22.0,
        "people_present": True, "num_people": 2, "step": 2
    })
    assert dec_next.selected_action in [ACTION_HEAT, ACTION_OFF], f"Direct reversal occurred: got {dec_next.selected_action}"
    assert dec_next.selected_action != ACTION_COOL, "Direct HEAT -> COOL reversal is forbidden"
    print("✓ 1.3: Severe anti-reversal penalty prevents direct opposite switching.")

    # 1.4 Closed-loop stability: 30 steps with target 20.0°C, initial 20.7°C
    env_hunt = RoomEnvironment(
        initial_temp=20.7, desired_temp=20.0, outside_temp=26.0,
        comfort_tolerance=1.5, people_present=True, num_people=2, random_seed=42
    )
    u_hunt_agent = UtilityBasedAgent()
    u_actions = []
    for s in range(1, 31):
        obs = env_hunt.get_state()
        dec = u_hunt_agent.decide_action(obs)
        env_hunt.step(dec.selected_action)
        u_actions.append(dec.selected_action)

    # Verify no chattering: consecutive opposite reversals (HEAT <-> COOL) must be 0
    direct_reversals = sum(
        1 for i in range(1, len(u_actions))
        if (u_actions[i] == ACTION_HEAT and u_actions[i-1] == ACTION_COOL) or
           (u_actions[i] == ACTION_COOL and u_actions[i-1] == ACTION_HEAT)
    )
    assert direct_reversals == 0, f"Detected {direct_reversals} direct reversals!"
    print(f"✓ 1.4: 30-step closed loop near target executed stably with 0 direct reversals.")

    # =========================================================================
    # PART 2: OCCUPANCY-AWARE UTILITY DECISION TESTS
    # =========================================================================
    print("\n--- PART 2: OCCUPANCY-AWARE UTILITY DECISION-MAKING ---")

    # 2.1 Empty room (0 occupants) during warm weather: must NOT activate cooling
    u_occ_agent = UtilityBasedAgent()
    obs_unocc_warm = {
        "room_temp": 24.5,
        "desired_temp": 22.0,
        "outside_temp": 33.0,
        "insulation_level": "Medium",
        "comfort_tolerance": 1.5,
        "people_present": False,
        "num_people": 0,
        "room_size": 25.0,
        "step": 1,
    }
    dec_unocc = u_occ_agent.decide_action(obs_unocc_warm)
    assert dec_unocc.selected_action == ACTION_OFF, (
        f"Empty room unnecessarily cooled! Expected OFF, got {dec_unocc.selected_action}."
    )
    assert "unoccupied" in dec_unocc.reason.lower()
    print("✓ 2.1: Empty room (0 occupants) at 24.5°C remains OFF to conserve energy.")

    # 2.2 Occupied room (3 occupants) under same conditions: MUST activate cooling
    u_occ_agent.reset()
    obs_occ_warm = {
        "room_temp": 24.5,
        "desired_temp": 22.0,
        "outside_temp": 33.0,
        "insulation_level": "Medium",
        "comfort_tolerance": 1.5,
        "people_present": True,
        "num_people": 3,
        "room_size": 25.0,
        "step": 1,
    }
    dec_occ = u_occ_agent.decide_action(obs_occ_warm)
    assert dec_occ.selected_action == ACTION_COOL, (
        f"Occupied room failed to cool! Expected COOL, got {dec_occ.selected_action}."
    )
    assert "occupied by 3" in dec_occ.reason.lower()
    print("✓ 2.2: Occupied room (3 people) at 24.5°C actively cools to maintain comfort.")

    # 2.3 Empty room under critical overheat conditions (> 35°C): Safety COOL permitted
    u_occ_agent.reset()
    obs_unocc_extreme = {
        "room_temp": 36.5,
        "desired_temp": 22.0,
        "outside_temp": 42.0,
        "insulation_level": "Medium",
        "comfort_tolerance": 1.5,
        "people_present": False,
        "num_people": 0,
        "room_size": 25.0,
        "step": 1,
    }
    dec_unocc_safety = u_occ_agent.decide_action(obs_unocc_extreme)
    assert dec_unocc_safety.selected_action == ACTION_COOL, (
        f"Critical overheat in empty room must activate safety COOL, got {dec_unocc_safety.selected_action}."
    )
    print("✓ 2.3: Empty room under extreme heat (>35°C) activates safety COOL.")

    # 2.4 Empty room under critical freeze conditions (< 12°C): Safety HEAT permitted
    u_occ_agent.reset()
    obs_unocc_freeze = {
        "room_temp": 10.5,
        "desired_temp": 22.0,
        "outside_temp": 0.0,
        "insulation_level": "Medium",
        "comfort_tolerance": 1.5,
        "people_present": False,
        "num_people": 0,
        "room_size": 25.0,
        "step": 1,
    }
    dec_freeze_safety = u_occ_agent.decide_action(obs_unocc_freeze)
    assert dec_freeze_safety.selected_action == ACTION_HEAT, (
        f"Critical freeze in empty room must activate safety HEAT, got {dec_freeze_safety.selected_action}."
    )
    print("✓ 2.4: Empty room under extreme freeze (<12°C) activates safety HEAT.")

    # 2.5 Dynamic Occupancy Transition: People Enter (0 -> 4 People) and Leave (4 -> 0 People)
    print("\n--- PART 3: DYNAMIC OCCUPANCY SIMULATION ---")
    dyn_env = RoomEnvironment(
        initial_temp=23.0, desired_temp=22.0, outside_temp=33.0,
        comfort_tolerance=1.5, people_present=False, num_people=0, random_seed=101
    )
    dyn_agent = UtilityBasedAgent()

    # Steps 1 to 10: Empty room -> must stay OFF
    unocc_actions = []
    for step in range(1, 11):
        obs = dyn_env.get_state()
        dec = dyn_agent.decide_action(obs)
        dyn_env.step(dec.selected_action)
        unocc_actions.append(dec.selected_action)
    assert unocc_actions.count(ACTION_OFF) >= 9, f"Unoccupied room ran active HVAC unnecessarily: {unocc_actions}"

    # Step 11: 4 People enter the room
    dyn_env.people_present = True
    dyn_env.num_people = 4

    occ_actions = []
    for step in range(11, 26):
        obs = dyn_env.get_state()
        dec = dyn_agent.decide_action(obs)
        dyn_env.step(dec.selected_action)
        occ_actions.append(dec.selected_action)
    # With 4 people and outdoor 33°C, cooling should activate to counteract body heat
    assert ACTION_COOL in occ_actions, "Agent did not cool when 4 occupants entered"

    # Step 26: People leave the room
    dyn_env.people_present = False
    dyn_env.num_people = 0

    leave_actions = []
    for step in range(26, 41):
        obs = dyn_env.get_state()
        dec = dyn_agent.decide_action(obs)
        dyn_env.step(dec.selected_action)
        leave_actions.append(dec.selected_action)
    # Once room is empty again, agent must relax to OFF setback mode
    assert leave_actions[-1] == ACTION_OFF, "Agent failed to revert to OFF after occupants left"
    print("✓ 2.5: Dynamic occupancy transition (0 -> 4 -> 0 people) handled smoothly without chattering.")

    # =========================================================================
    # PART 4: SYNCHRONIZED COMPARISON FAIRNESS & FULL STEP EXECUTION
    # =========================================================================
    print("\n--- PART 4: SYNCHRONIZED COMPARISON & FULL STEP EXECUTION ---")
    cfg_comp = ComparisonConfig(
        initial_temp=22.0,
        desired_temp=22.0,
        outside_temp=33.0,
        people_present=False,
        num_people=0,
        max_steps=50,
        random_seed=42,
    )
    runner = ComparisonRunner(cfg_comp)
    runner.run_all()
    eval_res = runner.evaluate()

    u_met = eval_res["utility_metrics"]
    r_met = eval_res["reflex_metrics"]

    # All 50 steps executed (no early exit)
    assert u_met.total_steps == 50, f"Utility steps: {u_met.total_steps} != 50"
    assert r_met.total_steps == 50, f"Reflex steps: {r_met.total_steps} != 50"

    # Energy in empty room: Utility agent must be much more efficient via setback drift
    assert u_met.total_energy < r_met.total_energy, (
        f"Utility energy ({u_met.total_energy}) should be less than Reflex energy ({r_met.total_energy}) in empty room."
    )
    # Direct reversals must be 0 for both
    assert u_met.unnecessary_action_changes == 0, f"Utility had {u_met.unnecessary_action_changes} unnecessary changes"
    assert r_met.unnecessary_action_changes == 0, f"Reflex had {r_met.unnecessary_action_changes} unnecessary changes"

    print(f"✓ 4.1: Synchronized Comparison completed all 50 steps:")
    print(f"       Utility Energy: {u_met.total_energy} units (Zero reversals)")
    print(f"       Reflex Energy:  {r_met.total_energy} units (Zero reversals)")

    print("\n==================================================================")
    print("ALL VERIFICATION SUITE TESTS PASSED SUCCESSFULLY (100% OK)!")
    print("==================================================================")
    return True


if __name__ == "__main__":
    ok = run_all_verification_tests()
    sys.exit(0 if ok else 1)
