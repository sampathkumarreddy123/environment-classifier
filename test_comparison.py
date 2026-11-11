"""
Verification suite for Phase 4: Synchronized Agent Comparison & Performance Evaluation.

Test Scenarios (Section 10):
1. Cold room with hot outside temperature.
2. Hot room with cold outside temperature.
3. Comfortable initial room temperature.
4. Poor insulation.
5. Good insulation.
6. Different desired temperatures.

Also verifies:
- Identical initial conditions and identical step-by-step disturbance sequences.
- Re-run determinism with the same random seed and pause/resume equivalence.
- Mathematical accuracy of all Temperature Control, Energy Efficiency, and Stability metrics.
- Environment classification summaries (Cold, Comfortable, Hot).
- Non-biased per-metric winner selection (including scenarios where Reflex wins).
- Zero-step and zero-energy edge cases and CSV export formatting.
"""

import io
import sys
import pandas as pd

from comparison import (
    AGENT_REFLEX,
    AGENT_UTILITY,
    TIE_LABEL,
    ComparisonConfig,
    ComparisonRunner,
    compute_agent_metrics,
    determine_metric_winner,
    determine_settling_winner,
)
from config import CLASS_COLD, CLASS_COMFORTABLE, CLASS_HOT


def _verify_runner_invariants(runner: ComparisonRunner, expected_steps: int) -> dict:
    """Assert shared environment conditions, identical disturbances, and valid CSV exports."""
    res = runner.evaluate()
    u_df = res["utility_df"]
    r_df = res["reflex_df"]
    combined_df = res["combined_df"]
    comp_df = res["comparison_df"]
    u_m = res["utility_metrics"]
    r_m = res["reflex_metrics"]

    # 1. Same number of rows (step 0 + expected_steps)
    assert len(u_df) == expected_steps + 1
    assert len(r_df) == expected_steps + 1
    assert len(combined_df) == expected_steps + 1

    # 2. Identical initial state & shared environmental conditions
    assert u_df.loc[0, "room_temp"] == r_df.loc[0, "room_temp"]
    assert u_df["outside_temp"].equals(r_df["outside_temp"])
    assert u_df["desired_temp"].equals(r_df["desired_temp"])
    assert u_df["disturbance"].equals(r_df["disturbance"]), (
        "Both agents must receive identical disturbances at every time step."
    )

    # 3. Classification sum checks
    assert u_m.cold_steps + u_m.comfortable_steps + u_m.hot_steps == expected_steps
    assert r_m.cold_steps + r_m.comfortable_steps + r_m.hot_steps == expected_steps
    assert all(
        c in {CLASS_COLD, CLASS_COMFORTABLE, CLASS_HOT}
        for c in u_df["classification"].tolist() + r_df["classification"].tolist()
    )

    # 4. Comparison table columns & valid winner entries
    expected_cols = [
        "Performance Metric",
        "Utility-Based Agent",
        "Simple Reflex Agent",
        "Better Performing Agent",
    ]
    assert list(comp_df.columns) == expected_cols
    assert len(comp_df) == 11
    assert set(comp_df["Better Performing Agent"]).issubset(
        {AGENT_UTILITY, AGENT_REFLEX, TIE_LABEL}
    )

    # 5. CSV round-trip checks
    for df in (u_df, r_df, combined_df, comp_df):
        csv_text = df.to_csv(index=False)
        parsed = pd.read_csv(io.StringIO(csv_text))
        assert len(parsed) == len(df)

    return res


def run_all_tests() -> bool:
    print("==========================================================")
    print("Running Phase 4 Synchronized Comparison Verification Suite")
    print("==========================================================")

    # -------------------------------------------------------------------------
    # Scenarios 1–6: Required Environmental Test Matrix
    # -------------------------------------------------------------------------
    scenarios = [
        (
            "1. Cold room with hot outside temperature",
            ComparisonConfig(
                initial_temp=13.0,
                desired_temp=22.0,
                outside_temp=38.0,
                insulation_level="Medium",
                max_steps=40,
                random_seed=42,
            ),
        ),
        (
            "2. Hot room with cold outside temperature",
            ComparisonConfig(
                initial_temp=34.0,
                desired_temp=22.0,
                outside_temp=4.0,
                insulation_level="Medium",
                max_steps=40,
                random_seed=42,
            ),
        ),
        (
            "3. Comfortable initial room temperature",
            ComparisonConfig(
                initial_temp=22.0,
                desired_temp=22.0,
                outside_temp=28.0,
                insulation_level="Medium",
                max_steps=35,
                random_seed=7,
            ),
        ),
        (
            "4. Poor insulation",
            ComparisonConfig(
                initial_temp=16.0,
                desired_temp=22.0,
                outside_temp=6.0,
                insulation_level="Poor",
                max_steps=40,
                random_seed=99,
            ),
        ),
        (
            "5. Good insulation",
            ComparisonConfig(
                initial_temp=16.0,
                desired_temp=22.0,
                outside_temp=6.0,
                insulation_level="Good",
                max_steps=40,
                random_seed=99,
            ),
        ),
        (
            "6a. Different desired temperature (19°C)",
            ComparisonConfig(
                initial_temp=24.0,
                desired_temp=19.0,
                outside_temp=28.0,
                insulation_level="Medium",
                lower_threshold=17.5,
                upper_threshold=20.5,
                max_steps=35,
                random_seed=11,
            ),
        ),
        (
            "6b. Different desired temperature (25°C)",
            ComparisonConfig(
                initial_temp=18.0,
                desired_temp=25.0,
                outside_temp=12.0,
                insulation_level="Medium",
                lower_threshold=23.5,
                upper_threshold=26.5,
                max_steps=35,
                random_seed=11,
            ),
        ),
    ]

    for label, cfg in scenarios:
        runner = ComparisonRunner(cfg)
        runner.run_all()
        res = _verify_runner_invariants(runner, cfg.max_steps)
        u_m = res["utility_metrics"]
        r_m = res["reflex_metrics"]
        print(
            f"✓ {label}: "
            f"Utility(MAE={u_m.avg_abs_error:.2f}°C, Comf={u_m.comfort_percentage:.1f}%, Energy={u_m.total_energy:.1f}) | "
            f"Reflex(MAE={r_m.avg_abs_error:.2f}°C, Comf={r_m.comfort_percentage:.1f}%, Energy={r_m.total_energy:.1f}) | "
            f"Summary={res['summary']['overall_winner']}"
        )

    # -------------------------------------------------------------------------
    # Test 7: Determinism (Re-run Same Seed) & Pause/Resume Equivalence
    # -------------------------------------------------------------------------
    cfg_det = ComparisonConfig(
        initial_temp=17.0,
        desired_temp=22.0,
        outside_temp=34.0,
        disturbance_level=0.35,
        max_steps=30,
        random_seed=1234,
    )
    r_full = ComparisonRunner(cfg_det)
    r_full.run_all()
    df_full = r_full.get_combined_df()

    # Re-run with same seed
    r_full.run_all()
    assert r_full.get_combined_df().equals(df_full), "Same-seed re-run must be identical."

    # Pause after 12 steps, then resume remaining 18 steps
    r_paused = ComparisonRunner(cfg_det)
    for _ in range(12):
        assert r_paused.step_once() is True
    assert r_paused.current_step == 12
    assert not r_paused.is_finished
    r_paused.run_remaining()
    assert r_paused.current_step == 30
    assert r_paused.is_finished
    assert r_paused.get_combined_df().equals(df_full), (
        "Paused & resumed run must match full uninterrupted run identically."
    )
    print("✓ Test 7 Passed: Same-seed determinism and Pause/Resume equivalence verified.")

    # -------------------------------------------------------------------------
    # Test 8: Exact Metric Math Verification & Edge Cases (Zero Energy, Zero Steps)
    # -------------------------------------------------------------------------
    synthetic_df = pd.DataFrame([
        {"step": 0, "room_temp": 18.0, "desired_temp": 22.0, "lower_comfort": 20.5, "upper_comfort": 23.5, "action": "OFF", "actual_temp": 18.0, "energy_consumed": 0.0, "classification": CLASS_COLD},
        {"step": 1, "room_temp": 18.0, "desired_temp": 22.0, "lower_comfort": 20.5, "upper_comfort": 23.5, "action": "HEAT", "actual_temp": 21.0, "energy_consumed": 1.0, "classification": CLASS_COMFORTABLE},
        {"step": 2, "room_temp": 21.0, "desired_temp": 22.0, "lower_comfort": 20.5, "upper_comfort": 23.5, "action": "OFF", "actual_temp": 24.0, "energy_consumed": 0.0, "classification": CLASS_HOT},  # overshoot above 23.5!
        {"step": 3, "room_temp": 24.0, "desired_temp": 22.0, "lower_comfort": 20.5, "upper_comfort": 23.5, "action": "HEAT", "actual_temp": 22.0, "energy_consumed": 1.0, "classification": CLASS_COMFORTABLE},  # chatter HEAT->OFF->HEAT!
        {"step": 4, "room_temp": 22.0, "desired_temp": 22.0, "lower_comfort": 20.5, "upper_comfort": 23.5, "action": "OFF", "actual_temp": 22.0, "energy_consumed": 0.0, "classification": CLASS_COMFORTABLE},  # chatter OFF->HEAT->OFF!
    ])
    m = compute_agent_metrics(synthetic_df)
    assert m.total_steps == 4
    # Errors across steps 1..4: |21-22|=1, |24-22|=2, |22-22|=0, |22-22|=0 -> mean = 3/4 = 0.75
    assert m.avg_abs_error == 0.75
    assert m.final_temp_error == 0.0
    # Comfort steps: steps 1, 3, 4 -> 3/4 = 75.0%
    assert m.comfort_steps == 3
    assert m.comfort_percentage == 75.0
    assert m.overshoot_count == 1
    assert m.total_energy == 2.0
    assert m.avg_energy_per_step == 0.5
    assert m.heating_activations == 2
    assert m.cooling_activations == 0
    # Fluctuations across steps 0->4: |21-18|=3, |24-21|=3, |22-24|=2, |22-22|=0 -> mean = 8/4 = 2.0
    assert m.temp_fluctuations == 2.0
    # Unnecessary action changes: HEAT, OFF, HEAT, OFF -> 2 chattering cycles
    assert m.unnecessary_action_changes == 2
    # Settled from step 3 onward (steps 3 and 4 Comfortable)
    assert m.settling_steps == 3

    # Edge case: zero energy consumption across whole run
    zero_cfg = ComparisonConfig(
        initial_temp=22.0,
        desired_temp=22.0,
        outside_temp=22.0,
        disturbance_level=0.0,
        lower_threshold=20.0,
        upper_threshold=24.0,
        max_steps=10,
        random_seed=1,
    )
    zero_runner = ComparisonRunner(zero_cfg)
    zero_runner.run_all()
    zero_res = _verify_runner_invariants(zero_runner, 10)
    assert zero_res["utility_metrics"].total_energy == 0.0
    assert zero_res["reflex_metrics"].total_energy == 0.0
    assert zero_res["utility_metrics"].avg_energy_per_step == 0.0
    assert zero_res["reflex_metrics"].avg_energy_per_step == 0.0
    assert zero_res["summary"]["overall_winner"] == TIE_LABEL

    # Edge case: 0 steps executed yet
    pre_runner = ComparisonRunner(zero_cfg)
    pre_res = pre_runner.evaluate()
    assert pre_res["utility_metrics"].total_steps == 0
    assert pre_res["summary"]["overall_winner"] == TIE_LABEL

    # Non-biased winner check: scenario where Reflex outperforms Utility on comfort/error
    # (Utility configured with extreme weight_energy=5.0 and weight_comfort=0.1, refusing to heat!)
    reflex_favored_cfg = ComparisonConfig(
        initial_temp=15.0,
        desired_temp=22.0,
        outside_temp=15.0,
        disturbance_level=0.0,
        weight_comfort=0.1,
        weight_energy=5.0,
        weight_stability=0.0,
        weight_overshoot=0.0,
        lower_threshold=20.5,
        upper_threshold=23.5,
        max_steps=20,
        random_seed=42,
    )
    rf_runner = ComparisonRunner(reflex_favored_cfg)
    rf_runner.run_all()
    rf_res = _verify_runner_invariants(rf_runner, 20)
    assert determine_metric_winner(
        rf_res["utility_metrics"].avg_abs_error,
        rf_res["reflex_metrics"].avg_abs_error,
        higher_is_better=False,
    ) == AGENT_REFLEX, "Reflex must win MAE when Utility stays off due to high energy weight."
    assert determine_settling_winner(None, 5) == AGENT_REFLEX
    assert determine_settling_winner(5, None) == AGENT_UTILITY
    assert determine_settling_winner(None, None) == TIE_LABEL

    print("✓ Test 8 Passed: Exact metric math, zero-energy/zero-step edge cases, and unbiased winner logic verified.")
    print("==========================================================")
    print("ALL PHASE 4 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("==========================================================")
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

