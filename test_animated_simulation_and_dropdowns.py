"""
Unit and regression tests for dropdown visibility styling, removal of unnecessary sidebar controls,
fixed progressive simulation delay, reflex initial action evaluation, and integer chart ticks.
"""

import unittest
import pandas as pd
from config import SimulationDefaults, CLASS_COLD, CLASS_COMFORTABLE, CLASS_HOT, ACTION_HEAT, ACTION_COOL, ACTION_OFF
from environment import RoomEnvironment
from utility_agent import UtilityBasedAgent
from reflex_agent import SimpleReflexAgent
from comparison import ComparisonConfig, ComparisonRunner
from utils import create_comparison_temperature_chart, create_agent_temperature_history_chart, create_reflex_temperature_history_chart


class TestAnimatedSimulationAndDropdowns(unittest.TestCase):
    """Test suite for updated UI, removed playback/auto-stop controls, reflex initial action, and chart ticks."""

    def test_dropdown_dark_and_light_css_colors(self):
        """Verify app.py contains the exact required Dark Mode and Light Mode colors and CSS rules."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        # Dark Mode colors
        self.assertIn('"#172B24"', code, "Dark Mode dropdown background missing")
        self.assertIn('"#E8F1E9"', code, "Dark Mode primary text missing")
        self.assertIn('"#B5C9BA"', code, "Dark Mode secondary text missing")
        self.assertIn('"#28543F"', code, "Dark Mode selected option background missing")
        self.assertIn('"#FFFFFF"', code, "Dark Mode selected option text missing")
        self.assertIn('"#244A38"', code, "Dark Mode hover background missing")
        self.assertIn('"#456B55"', code, "Dark Mode border missing")
        self.assertIn('"#72B58A"', code, "Dark Mode focus outline missing")

        # Light Mode colors
        self.assertIn('"#FFFFFF"', code, "Light Mode dropdown background missing")
        self.assertIn('"#243B30"', code, "Light Mode option text missing")
        self.assertIn('"#E4F0E5"', code, "Light Mode selected background missing")
        self.assertIn('"#EDF5ED"', code, "Light Mode hover background missing")
        self.assertIn('"#D6E4D6"', code, "Light Mode border missing")

    def test_removal_of_unnecessary_sidebar_controls(self):
        """Verify Playback Speed and Auto-Stop at Target controls are completely removed from app.py."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        # Removed controls must NOT exist in app.py
        self.assertNotIn("sim_playback_speed", code, "sim_playback_speed still exists in app.py")
        self.assertNotIn("stop_at_target", code, "stop_at_target still exists in app.py")
        self.assertNotIn("Simulation Playback Speed", code, "Playback speed selector label still exists in app.py")
        self.assertNotIn("Stop When Target Reached", code, "Stop at target toggle label still exists in app.py")
        self.assertNotIn("PLAYBACK SPEED", code, "Playback speed section caption still exists in app.py")
        self.assertNotIn("AUTO-STOP AT TARGET", code, "Auto-stop section caption still exists in app.py")

        # Fixed visualization delay must be defined
        self.assertTrue("VISUALIZATION_DELAY_SEC" in code and "0.25" in code, "VISUALIZATION_DELAY_SEC missing in app.py")

    def test_reflex_agent_initial_action_evaluation(self):
        """Verify reflex agent evaluates condition-action rules at step 0 (e.g., 18°C < 20°C -> HEAT)."""
        # Lower threshold 20°C, initial temp 18°C
        cfg = ComparisonConfig(initial_temp=18.0, lower_threshold=20.0, upper_threshold=26.0)
        runner = ComparisonRunner(cfg)

        r_df = runner.get_reflex_df()
        self.assertFalse(r_df.empty)
        # Step 0 action should be HEAT, not OFF
        step_0_action = r_df["action"].iloc[0]
        self.assertEqual(step_0_action, ACTION_HEAT, f"Expected step 0 action HEAT for 18°C < 20°C, got {step_0_action}")

        # Test reflex agent preview_decision directly
        agent = SimpleReflexAgent(lower_threshold=20.0, upper_threshold=26.0)
        dec = agent.preview_decision({"room_temp": 18.0, "outside_temp": 10.0, "desired_temp": 22.0})
        self.assertEqual(dec.selected_action, ACTION_HEAT)

        dec_hot = agent.preview_decision({"room_temp": 28.0, "outside_temp": 32.0, "desired_temp": 22.0})
        self.assertEqual(dec_hot.selected_action, ACTION_COOL)

        dec_comf = agent.preview_decision({"room_temp": 22.0, "outside_temp": 22.0, "desired_temp": 22.0})
        self.assertEqual(dec_comf.selected_action, ACTION_OFF)

    def test_comparison_telemetry_columns_actual_temp(self):
        """Verify get_utility_df() and get_reflex_df() both contain actual_temp and room_temp."""
        cfg = ComparisonConfig(initial_temp=18.0, max_steps=5)
        runner = ComparisonRunner(cfg)
        runner.step_once()

        u_df = runner.get_utility_df()
        r_df = runner.get_reflex_df()

        self.assertIn("actual_temp", u_df.columns, "actual_temp missing in utility_df")
        self.assertIn("room_temp", u_df.columns, "room_temp missing in utility_df")
        self.assertIn("actual_temp", r_df.columns, "actual_temp missing in reflex_df")
        self.assertIn("room_temp", r_df.columns, "room_temp missing in reflex_df")

        # Also test RoomEnvironment.get_history_df()
        env = RoomEnvironment(initial_temp=18.0)
        env.step(ACTION_HEAT)
        env_df = env.get_history_df()
        self.assertIn("actual_temp", env_df.columns, "actual_temp missing in RoomEnvironment.get_history_df()")
        self.assertIn("room_temp", env_df.columns, "room_temp missing in RoomEnvironment.get_history_df()")

    def test_temperature_chart_integer_ticks(self):
        """Verify temperature charts enforce integer step tick formatting."""
        cfg = ComparisonConfig(initial_temp=18.0, max_steps=10)
        runner = ComparisonRunner(cfg)
        for _ in range(5):
            runner.step_once()

        u_df = runner.get_utility_df()
        r_df = runner.get_reflex_df()

        fig_comp = create_comparison_temperature_chart(utility_df=u_df, reflex_df=r_df)
        xaxis = fig_comp.layout.xaxis
        self.assertEqual(xaxis.tickformat, "d", "Chart x-axis tickformat should be 'd' (integer)")
        self.assertEqual(xaxis.tickmode, "linear", "Chart x-axis tickmode should be 'linear'")

        fig_agent = create_agent_temperature_history_chart(df=u_df)
        self.assertEqual(fig_agent.layout.xaxis.tickformat, "d")

        fig_refl = create_reflex_temperature_history_chart(df=r_df)
        self.assertEqual(fig_refl.layout.xaxis.tickformat, "d")

    def test_stepwise_synchronized_comparison_progression(self):
        """Verify that step_once executes synchronized steps identically to run_all."""
        cfg = ComparisonConfig(
            initial_temp=14.0,
            desired_temp=22.0,
            outside_temp=10.0,
            max_steps=10,
            disturbance_level=0.5,
            random_seed=42,
        )

        # 1. Run via stepwise loop
        step_runner = ComparisonRunner(cfg)
        step_count = 0
        while not step_runner.is_finished:
            success = step_runner.step_once()
            if not success:
                break
            step_count += 1
            # Verify synchronous steps
            self.assertEqual(step_runner.util_env.current_step, step_count)
            self.assertEqual(step_runner.refl_env.current_step, step_count)
            self.assertEqual(len(step_runner.utility_records), step_count + 1)
            self.assertEqual(len(step_runner.reflex_records), step_count + 1)

        self.assertEqual(step_count, 10)
        stepwise_eval = step_runner.evaluate()

        # 2. Run via run_all
        all_runner = ComparisonRunner(cfg)
        all_runner.run_all()
        all_eval = all_runner.evaluate()

        # Both must produce identical results with identical seeds
        self.assertEqual(stepwise_eval["summary"]["overall_winner"], all_eval["summary"]["overall_winner"])
        self.assertAlmostEqual(stepwise_eval["utility_metrics"].avg_abs_error, all_eval["utility_metrics"].avg_abs_error, places=3)
        self.assertAlmostEqual(stepwise_eval["reflex_metrics"].avg_abs_error, all_eval["reflex_metrics"].avg_abs_error, places=3)
        self.assertAlmostEqual(stepwise_eval["utility_metrics"].total_energy, all_eval["utility_metrics"].total_energy, places=2)
        self.assertAlmostEqual(stepwise_eval["reflex_metrics"].total_energy, all_eval["reflex_metrics"].total_energy, places=2)

    def test_cooling_and_heating_stepwise_actions(self):
        """Verify that stepwise simulation actions transition correctly under heating and cooling scenarios."""
        # Cold scenario with occupants
        env_cold = RoomEnvironment(initial_temp=14.0, desired_temp=22.0, outside_temp=8.0, people_present=True, num_people=2)
        agent = UtilityBasedAgent()
        obs = env_cold.get_state()
        dec = agent.decide_action(obs)
        self.assertEqual(dec.selected_action, ACTION_HEAT)

        next_st = env_cold.step(dec.selected_action)
        self.assertGreater(next_st["room_temp"], 14.0)

        # Hot scenario with occupants
        env_hot = RoomEnvironment(initial_temp=30.0, desired_temp=22.0, outside_temp=32.0, people_present=True, num_people=2)
        obs_h = env_hot.get_state()
        dec_h = agent.decide_action(obs_h)
        self.assertEqual(dec_h.selected_action, ACTION_COOL)

        next_st_h = env_hot.step(dec_h.selected_action)
        self.assertLess(next_st_h["room_temp"], 30.0)

        # Comfortable scenario with occupants
        env_comf = RoomEnvironment(initial_temp=22.0, desired_temp=22.0, outside_temp=22.0, people_present=True, num_people=2)
        obs_c = env_comf.get_state()
        dec_c = agent.decide_action(obs_c)
        self.assertEqual(dec_c.selected_action, ACTION_OFF)

        # Unoccupied scenario: all machines must be OFF when there are no people
        env_unocc = RoomEnvironment(initial_temp=14.0, desired_temp=22.0, outside_temp=8.0, people_present=False, num_people=0)
        obs_u = env_unocc.get_state()
        dec_u = agent.decide_action(obs_u)
        self.assertEqual(dec_u.selected_action, ACTION_OFF)

    def test_live_sim_agent_buttons_and_fan_window_removed(self):
        """Verify Ceiling Fan Mode and Natural Ventilation controls are removed and agent buttons are side by side."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        # Ceiling Fan Mode and Natural Ventilation controls must NOT exist
        self.assertNotIn("sim_fan_mode_select", code)
        self.assertNotIn("sim_window_mode_select", code)
        self.assertNotIn('"Ceiling Fan Mode"', code)
        self.assertNotIn('"Natural Ventilation"', code)

        # Controlling agent radio must be horizontal
        self.assertIn('key="sim_active_agent_radio"', code)
        self.assertIn('horizontal=True', code)


if __name__ == "__main__":
    unittest.main()
