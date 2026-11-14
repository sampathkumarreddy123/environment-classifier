"""
Unit and regression tests for dropdown visibility styling and real-time gradual simulation playback.
"""

import unittest
import pandas as pd
from config import SimulationDefaults, CLASS_COLD, CLASS_COMFORTABLE, CLASS_HOT, ACTION_HEAT, ACTION_COOL, ACTION_OFF
from environment import RoomEnvironment
from utility_agent import UtilityBasedAgent
from reflex_agent import SimpleReflexAgent
from comparison import ComparisonConfig, ComparisonRunner


class TestAnimatedSimulationAndDropdowns(unittest.TestCase):
    """Test suite for dropdown styling verification and stepwise animated simulation consistency."""

    def test_dropdown_dark_and_light_css_colors(self):
        """Verify app.py contains the exact required Dark Mode and Light Mode colors and CSS rules."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        # Dark Mode colors
        self.assertIn('"#172B24"', code, "Dark Mode dropdown background #172B24 missing")
        self.assertIn('"#28543F"', code, "Dark Mode selected option background #28543F missing")
        self.assertIn('"#456B55"', code, "Dark Mode dropdown border #456B55 missing")
        self.assertIn('"#72B58A"', code, "Dark Mode focus outline #72B58A missing")
        self.assertIn('"#E8F1E9"', code, "Dark Mode primary text #E8F1E9 missing")

        # Light Mode colors
        self.assertIn('"#FFFFFF"', code, "Light Mode dropdown background #FFFFFF missing")
        self.assertIn('"#E4F0E5"', code, "Light Mode selected option background #E4F0E5 missing")
        self.assertIn('"#D6E4D6"', code, "Light Mode border #D6E4D6 missing")
        self.assertIn('"#245C43"', code, "Light Mode focus outline #245C43 missing")
        self.assertIn('"#243B30"', code, "Light Mode option text #243B30 missing")

        # Popover visibility & z-index
        self.assertIn('z-index: 9999999 !important;', code, "Dropdown popover z-index missing")
        self.assertIn('word-break: break-word !important;', code, "Option word-break missing")
        self.assertIn('white-space: normal !important;', code, "Option white-space missing")

    def test_playback_speed_delays(self):
        """Verify playback speeds are properly defined in session state and mapping."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        self.assertIn('"Slow (500 ms/step)": 0.50', code)
        self.assertIn('"Normal (250 ms/step)": 0.25', code)
        self.assertIn('"Fast (100 ms/step)": 0.10', code)
        self.assertIn('"sim_playback_speed": "Normal (250 ms/step)"', code)

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
        # Cold scenario
        env_cold = RoomEnvironment(initial_temp=14.0, desired_temp=22.0, outside_temp=8.0)
        agent = UtilityBasedAgent()
        obs = env_cold.get_state()
        dec = agent.decide_action(obs)
        self.assertEqual(dec.selected_action, ACTION_HEAT)

        next_st = env_cold.step(dec.selected_action)
        self.assertGreater(next_st["room_temp"], 14.0)

        # Hot scenario
        env_hot = RoomEnvironment(initial_temp=30.0, desired_temp=22.0, outside_temp=32.0)
        obs_h = env_hot.get_state()
        dec_h = agent.decide_action(obs_h)
        self.assertEqual(dec_h.selected_action, ACTION_COOL)

        next_st_h = env_hot.step(dec_h.selected_action)
        self.assertLess(next_st_h["room_temp"], 30.0)

        # Comfortable scenario
        env_comf = RoomEnvironment(initial_temp=22.0, desired_temp=22.0, outside_temp=22.0)
        obs_c = env_comf.get_state()
        dec_c = agent.decide_action(obs_c)
        self.assertEqual(dec_c.selected_action, ACTION_OFF)


if __name__ == "__main__":
    unittest.main()
