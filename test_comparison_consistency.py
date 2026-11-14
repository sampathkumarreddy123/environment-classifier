"""
End-to-End Simulation Data Consistency and Comparison Engine Verification.
Tests the exact fixes for:
1. Unsimulated (step 0) state: no fabricated 4°C error, winner = "Simulation Not Run", total_steps = 0.
2. Synchronized results shared between Agent Comparison and Results page.
3. Multiple occupancy conditions (0, 1, 6 people) and room sizes (12m², 25m², 60m²).
4. Reset and re-run consistency without state leakage.
"""

import sys
import pandas as pd
from comparison import (
    ComparisonConfig,
    ComparisonRunner,
    compute_agent_metrics,
    build_comparison_table_df,
    summarize_comparison,
    AGENT_UTILITY,
    AGENT_REFLEX,
    TIE_LABEL,
)


def test_unsimulated_state():
    print("Testing unsimulated (step 0) state...")
    runner = ComparisonRunner(ComparisonConfig())
    eval_res = runner.evaluate()
    
    u_metrics = eval_res["utility_metrics"]
    r_metrics = eval_res["reflex_metrics"]
    comp_df = eval_res["comparison_df"]
    summary = eval_res["summary"]
    
    assert u_metrics.total_steps == 0, f"Expected 0 steps, got {u_metrics.total_steps}"
    assert r_metrics.total_steps == 0, f"Expected 0 steps, got {r_metrics.total_steps}"
    assert u_metrics.avg_abs_error == 0.0, f"Expected 0.0 error, got {u_metrics.avg_abs_error}"
    assert r_metrics.avg_abs_error == 0.0, f"Expected 0.0 error, got {r_metrics.avg_abs_error}"
    assert u_metrics.total_energy == 0.0
    assert r_metrics.total_energy == 0.0
    
    # Check that table shows N/A (Not Run) and no fabricated winner
    assert comp_df.loc[comp_df["Performance Metric"] == "Average Absolute Temperature Error (°C)", "Utility-Based Agent"].values[0] == "N/A (Not Run)"
    assert comp_df.loc[comp_df["Performance Metric"] == "Average Absolute Temperature Error (°C)", "Better Performing Agent"].values[0] == "None (Simulation Not Run)"
    
    assert summary["overall_winner"] == "Simulation Not Run", f"Expected 'Simulation Not Run', got {summary['overall_winner']}"
    assert summary["utility_wins"] == 0
    assert summary["reflex_wins"] == 0
    assert summary["ties"] == 0
    print("✓ Unsimulated state verified: No fabricated 4°C errors and winner is 'Simulation Not Run'.")


def test_completed_simulation_consistency():
    print("Testing completed simulation consistency...")
    cfg = ComparisonConfig(
        initial_temp=18.0,
        desired_temp=22.0,
        outside_temp=30.0,
        max_steps=40,
        random_seed=42,
    )
    runner = ComparisonRunner(cfg)
    runner.run_all()
    eval_res = runner.evaluate()
    
    u_metrics = eval_res["utility_metrics"]
    r_metrics = eval_res["reflex_metrics"]
    comp_df = eval_res["comparison_df"]
    summary = eval_res["summary"]
    comb_df = eval_res["combined_df"]
    
    assert len(comb_df) == 41  # step 0 to 40
    assert u_metrics.total_steps == 40
    assert r_metrics.total_steps == 40
    assert u_metrics.avg_abs_error > 0.0
    assert r_metrics.avg_abs_error > 0.0
    assert u_metrics.total_energy > 0.0 or r_metrics.total_energy > 0.0
    
    # Table should have 14 metrics with real values
    assert len(comp_df) == 14
    assert comp_df["Better Performing Agent"].str.contains("N/A|None").sum() == 0
    
    # Summary winner should strictly equal calculation
    winners = comp_df["Better Performing Agent"].tolist()
    u_w = winners.count(AGENT_UTILITY)
    r_w = winners.count(AGENT_REFLEX)
    t_w = winners.count(TIE_LABEL)
    assert summary["utility_wins"] == u_w
    assert summary["reflex_wins"] == r_w
    assert summary["ties"] == t_w
    if u_w > r_w:
        assert summary["overall_winner"] == AGENT_UTILITY
    elif r_w > u_w:
        assert summary["overall_winner"] == AGENT_REFLEX
    else:
        assert summary["overall_winner"] == TIE_LABEL
        
    print(f"✓ Completed simulation verified: 40 steps, Utility wins={u_w}, Reflex wins={r_w}, Ties={t_w}, Winner={summary['overall_winner']}.")


def test_occupancy_and_room_size_scenarios():
    print("Testing occupancy and room size scenarios...")
    test_cases = [
        ("Empty Room (0 Occupants)", ComparisonConfig(people_present=False, num_people=0, room_size=25.0)),
        ("1 Occupant", ComparisonConfig(people_present=True, num_people=1, room_size=25.0)),
        ("Multiple Occupants (6 People)", ComparisonConfig(people_present=True, num_people=6, room_size=25.0)),
        ("Small Room (12 m²)", ComparisonConfig(people_present=True, num_people=2, room_size=12.0)),
        ("Large Room (60 m²)", ComparisonConfig(people_present=True, num_people=2, room_size=60.0)),
    ]
    
    for name, cfg in test_cases:
        runner = ComparisonRunner(cfg)
        runner.run_all()
        res = runner.evaluate()
        u_m = res["utility_metrics"]
        r_m = res["reflex_metrics"]
        comb_df = res["combined_df"]
        
        assert u_m.total_steps == cfg.max_steps
        assert r_m.total_steps == cfg.max_steps
        assert not comb_df.isna().any().any(), f"NaN found in {name}!"
        print(f"  ✓ {name}: Evaluated {cfg.max_steps} steps without errors or NaNs.")
    print("✓ All occupancy and room size scenarios verified.")


def test_reset_and_rerun_lifecycle():
    print("Testing reset and re-run lifecycle...")
    runner = ComparisonRunner(ComparisonConfig(max_steps=20))
    runner.run_all()
    res1 = runner.evaluate()
    assert res1["utility_metrics"].total_steps == 20
    
    # Reset
    runner.reset()
    res_reset = runner.evaluate()
    assert res_reset["utility_metrics"].total_steps == 0
    assert res_reset["summary"]["overall_winner"] == "Simulation Not Run"
    
    # Re-run
    runner.run_all()
    res2 = runner.evaluate()
    assert res2["utility_metrics"].total_steps == 20
    assert res2["utility_metrics"].avg_abs_error == res1["utility_metrics"].avg_abs_error
    print("✓ Reset and re-run lifecycle verified identically.")


if __name__ == "__main__":
    test_unsimulated_state()
    test_completed_simulation_consistency()
    test_occupancy_and_room_size_scenarios()
    test_reset_and_rerun_lifecycle()
    print("==================================================")
    print("ALL DATA CONSISTENCY & INTEGRATION TESTS PASSED!")
    print("==================================================")
