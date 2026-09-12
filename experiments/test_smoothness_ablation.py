"""
Controlled smoothness ablation for the ReLU-vs-Tanh Anderson claim.

The historical experiments used different best learning rates and different
seed ranges for ReLU and Tanh. This script runs the same seed IDs and the same
learning-rate grid for both activations, so the activation-smoothness claim can
be treated as a controlled ablation rather than a tuned-condition comparison.
"""

import copy
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import get_digits_data
from src.models import SmallCNN, SmallCNNTanh
from src.provenance import write_experiment_result
from src.statistics import format_results_table, paired_analysis
from src.training import train_baseline, train_with_anderson


def run_activation_lr(model_fn, X, y, loss_fn, seeds, lr, steps, momentum, window, aa_interval):
    baseline_losses = []
    anderson_losses = []
    jump_counts = []

    for seed in seeds:
        torch.manual_seed(seed)
        model_b = model_fn()
        model_aa = copy.deepcopy(model_b)

        loss_b, _ = train_baseline(model_b, X, y, loss_fn, steps=steps, lr=lr, momentum=momentum)
        loss_aa, info = train_with_anderson(
            model_aa,
            X,
            y,
            loss_fn,
            steps=steps,
            lr=lr,
            momentum=momentum,
            window=window,
            aa_interval=aa_interval,
            safeguard=True,
        )
        baseline_losses.append(loss_b)
        anderson_losses.append(loss_aa)
        jump_counts.append(
            {
                "attempted": info["jumps_attempted"],
                "accepted": info["jumps_accepted"],
            }
        )

    return paired_analysis(baseline_losses, anderson_losses), {
        "baseline_losses": baseline_losses,
        "anderson_losses": anderson_losses,
        "jump_counts": jump_counts,
    }


def main(n_seeds=30, seed_start=7000, lrs=(0.03, 0.05, 0.1), steps=100, output_path=None):
    print("=" * 68)
    print("  Controlled Activation Smoothness Ablation")
    print("=" * 68)

    X, y = get_digits_data()
    loss_fn = nn.CrossEntropyLoss()
    seeds = range(seed_start, seed_start + n_seeds)
    configs = {
        "relu": SmallCNN,
        "tanh": SmallCNNTanh,
    }

    results = {}
    per_seed = {}
    for activation, model_fn in configs.items():
        for lr in lrs:
            key = f"{activation}_lr_{lr:g}"
            analysis, raw = run_activation_lr(
                model_fn,
                X,
                y,
                loss_fn,
                seeds,
                lr=lr,
                steps=steps,
                momentum=0.7,
                window=5,
                aa_interval=10,
            )
            results[key] = analysis
            per_seed[key] = raw
            print(format_results_table(analysis, title=f"{activation.upper()} lr={lr:g}: Baseline vs Anderson"))

    write_experiment_result(
        output_path or os.path.join("results", "test_smoothness_ablation_summary.json"),
        experiment="Same-seed same-lr activation smoothness ablation",
        config={
            "seeds": seeds,
            "learning_rates": lrs,
            "steps": steps,
            "momentum": 0.7,
            "window": 5,
            "aa_interval": 10,
            "dataset": "sklearn digits, full-batch training loss",
        },
        results=results,
        per_seed=per_seed,
        notes=[
            "Use this script to support causal activation-smoothness language.",
            "The historical ReLU and Tanh runs remain tuned-condition comparisons.",
        ],
    )
    print("=" * 68)
    return results


if __name__ == "__main__":
    main()
