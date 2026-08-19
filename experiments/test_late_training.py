"""
Late-Training Behavior Investigation (Test #3)

Investigates whether Anderson acceleration exhibits a late-stage performance
reversal or saturation as training extends far beyond initial convergence,
as observed in Ito & Xue (2025).

Protocol:
- Architecture: SmallCNNTanh on full-batch Digits dataset
- Horizons tested: T in {100, 250, 500, 1000} steps
- Sample size: 60 paired seeds (seeds 8000 - 8059)
- Metrics: Mean loss, relative reduction %, win/loss count, paired t-test p-value
- Trajectory tracking: Step-by-step loss logging to identify crossover dynamics
"""

import os
import sys
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import SmallCNNTanh
from src.data import get_digits_data
from src.training import train_baseline, train_with_anderson
from src.statistics import paired_analysis, format_results_table
from src.provenance import write_experiment_result


def run_late_training_experiment(horizons=(100, 250, 500, 1000), n_seeds=60, seed_start=8000):
    print("=" * 68)
    print("  Late-Training Behavior Investigation (Test #3)")
    print("=" * 68)

    X, y = get_digits_data()
    loss_fn = nn.CrossEntropyLoss()
    seeds = range(seed_start, seed_start + n_seeds)

    horizon_results = {}
    per_seed_results = {}
    detailed_trajectories = {"baseline": [], "anderson": []}

    for T in horizons:
        print(f"\n--- Testing Horizon T = {T} steps ({n_seeds} seeds) ---")
        baseline_losses = []
        anderson_losses = []

        for i, seed in enumerate(seeds):
            torch.manual_seed(seed)
            model_b = SmallCNNTanh()
            model_aa = copy.deepcopy(model_b)

            log_traj = (T == max(horizons) and i < 10)  # Log trajectories for first 10 seeds at longest horizon

            loss_b, info_b = train_baseline(
                model_b, X, y, loss_fn, steps=T, lr=0.1, momentum=0.7, log_trajectory=log_traj
            )
            loss_aa, info_aa = train_with_anderson(
                model_aa,
                X,
                y,
                loss_fn,
                steps=T,
                lr=0.1,
                momentum=0.7,
                window=5,
                aa_interval=10,
                reg=1e-6,
                safeguard=True,
                log_trajectory=log_traj,
            )

            baseline_losses.append(loss_b)
            anderson_losses.append(loss_aa)

            if log_traj:
                detailed_trajectories["baseline"].append(info_b["losses"])
                detailed_trajectories["anderson"].append(info_aa["losses"])

        res = paired_analysis(baseline_losses, anderson_losses)
        horizon_results[T] = res
        per_seed_results[T] = {
            "baseline_losses": baseline_losses,
            "anderson_losses": anderson_losses,
        }
        print(format_results_table(res, title=f"Horizon T = {T} Steps ({n_seeds} seeds)"))

    # Save summary dictionary
    os.makedirs("results", exist_ok=True)
    summary_path = os.path.join("results", "test_late_training_summary.json")
    with open(summary_path, "w") as f:
        # Convert non-serializable elements
        serializable = {
            k: {sk: (sv if not isinstance(sv, np.generic) else sv.item()) for sk, sv in v.items()}
            for k, v in horizon_results.items()
        }
        json.dump(serializable, f, indent=2)

    write_experiment_result(
        os.path.join("results", "test_late_training_provenance.json"),
        experiment="Late-training horizon dynamics",
        config={
            "horizons": horizons,
            "n_seeds": n_seeds,
            "seed_start": seed_start,
            "lr": 0.1,
            "momentum": 0.7,
            "window": 5,
            "aa_interval": 10,
            "model": "SmallCNNTanh",
        },
        results=horizon_results,
        per_seed=per_seed_results,
        notes=["Trajectory plots log the first 10 seeds at the longest horizon."],
    )

    # Plot Horizon Analysis
    plot_results(horizons, horizon_results, detailed_trajectories)

    return horizon_results


def plot_results(horizons, horizon_results, detailed_trajectories):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Loss Reduction vs. Horizon
    reductions = [horizon_results[T]["rel_reduction_pct"] for T in horizons]
    win_rates = [horizon_results[T]["wins"] / horizon_results[T]["n_seeds"] * 100 for T in horizons]

    color = 'tab:blue'
    axs[0].set_xlabel('Training Steps (Horizon T)', fontsize=11)
    axs[0].set_ylabel('Loss Reduction (%)', color=color, fontsize=11)
    line1 = axs[0].plot(horizons, reductions, 'o-', color=color, linewidth=2, label='Loss Reduction %')
    axs[0].tick_params(axis='y', labelcolor=color)
    axs[0].grid(True, alpha=0.3)

    ax0_twin = axs[0].twinx()
    color2 = 'tab:green'
    ax0_twin.set_ylabel('Win Rate (%)', color=color2, fontsize=11)
    line2 = ax0_twin.plot(horizons, win_rates, 's--', color=color2, linewidth=2, label='Win Rate %')
    ax0_twin.tick_params(axis='y', labelcolor=color2)
    ax0_twin.set_ylim([0, 105])

    axs[0].set_title('Anderson Advantage Across Training Horizons', fontsize=12)

    # 2. Step-by-step Loss Trajectory Average
    if len(detailed_trajectories["baseline"]) > 0:
        base_arr = np.array(detailed_trajectories["baseline"])
        aa_arr = np.array(detailed_trajectories["anderson"])
        mean_base = base_arr.mean(axis=0)
        mean_aa = aa_arr.mean(axis=0)
        steps = np.arange(len(mean_base))

        axs[1].semilogy(steps, mean_base, 'r-', label='Baseline SGD', alpha=0.8)
        axs[1].semilogy(steps, mean_aa, 'b-', label='Safeguarded Anderson', alpha=0.8)
        axs[1].set_xlabel('Step', fontsize=11)
        axs[1].set_ylabel('CrossEntropy Loss (log scale)', fontsize=11)
        axs[1].set_title('Mean Trajectory Evolution (10 Seeds)', fontsize=12)
        axs[1].grid(True, alpha=0.3, which='both')
        axs[1].legend()

    plt.tight_layout()
    plot_path = os.path.join("results", "test_late_training_dynamics.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nGenerated analysis figure saved to: {plot_path}")


def main():
    run_late_training_experiment(horizons=(100, 250, 500, 1000), n_seeds=60, seed_start=8000)


if __name__ == "__main__":
    main()
