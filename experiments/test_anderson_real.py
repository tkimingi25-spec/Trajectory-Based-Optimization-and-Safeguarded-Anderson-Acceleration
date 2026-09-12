"""
Real-Network Validation & Activation Smoothness Isolation (§5.11 - §5.13)

Demonstrates:
1. §5.11: TinyMLP regression on synthetic nonlinear data (30 seeds, ~29% loss reduction).
2. §5.12: SmallCNN (ReLU) on handwritten digits (60 seeds, robust null result p > 0.3).
3. §5.13: SmallCNNTanh (tanh) on identical CNN architecture & digits
   as the historical tuned-condition positive result.
"""

import copy
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import get_digits_data, get_tinymlp_data
from src.models import SmallCNN, SmallCNNTanh, TinyMLP
from src.provenance import write_experiment_result
from src.statistics import format_results_table, paired_analysis
from src.training import train_baseline, train_with_anderson


def test_tinymlp():
    print("\n--- 1. TinyMLP Regression Benchmark (§5.11, 30 seeds: 4000-4029) ---")
    seeds = range(4000, 4030)
    loss_fn = nn.MSELoss()
    baseline_losses = []
    anderson_losses = []

    for seed in seeds:
        X, y = get_tinymlp_data(N=200, seed=seed)
        torch.manual_seed(seed)
        model_b = TinyMLP()
        model_aa = copy.deepcopy(model_b)

        loss_b, _ = train_baseline(model_b, X, y, loss_fn, steps=150, lr=0.05, momentum=0.7)
        loss_aa, _ = train_with_anderson(
            model_aa, X, y, loss_fn, steps=150, lr=0.05, momentum=0.7, window=5, aa_interval=10
        )

        baseline_losses.append(loss_b)
        anderson_losses.append(loss_aa)

    res = paired_analysis(baseline_losses, anderson_losses)
    print(format_results_table(res, title="TinyMLP: Baseline vs Safeguarded Anderson (30 seeds)"))
    write_experiment_result(
        os.path.join("results", "test_anderson_real_tinymlp.json"),
        experiment="TinyMLP regression baseline vs safeguarded Anderson",
        config={"seeds": seeds, "steps": 150, "lr": 0.05, "momentum": 0.7, "window": 5, "aa_interval": 10},
        results=res,
        per_seed={"baseline_losses": baseline_losses, "anderson_losses": anderson_losses},
    )
    return res


def test_small_cnn_relu():
    print("\n--- 2. SmallCNN (ReLU) on Digits (§5.12, 60 seeds: 6000-6059) ---")
    seeds = range(6000, 6060)
    X, y = get_digits_data()
    loss_fn = nn.CrossEntropyLoss()
    baseline_losses = []
    anderson_losses = []

    for seed in seeds:
        torch.manual_seed(seed)
        model_b = SmallCNN()
        model_aa = copy.deepcopy(model_b)

        loss_b, _ = train_baseline(model_b, X, y, loss_fn, steps=100, lr=0.05, momentum=0.7)
        loss_aa, _ = train_with_anderson(
            model_aa, X, y, loss_fn, steps=100, lr=0.05, momentum=0.7, window=5, aa_interval=10
        )

        baseline_losses.append(loss_b)
        anderson_losses.append(loss_aa)

    res = paired_analysis(baseline_losses, anderson_losses)
    print(format_results_table(res, title="SmallCNN (ReLU): Baseline vs Anderson (60 seeds)"))
    write_experiment_result(
        os.path.join("results", "test_anderson_real_relu.json"),
        experiment="SmallCNN ReLU full-batch baseline vs safeguarded Anderson",
        config={"seeds": seeds, "steps": 100, "lr": 0.05, "momentum": 0.7, "window": 5, "aa_interval": 10},
        results=res,
        per_seed={"baseline_losses": baseline_losses, "anderson_losses": anderson_losses},
        notes=[
            "Use this as a ReLU null result for the tested hyperparameters, not as universal non-smooth activation proof."
        ],
    )
    return res


def test_small_cnn_tanh():
    print("\n--- 3. SmallCNNTanh (Smooth Activation) on Digits (§5.13, 60 seeds: 8000-8059) ---")
    seeds = range(8000, 8060)
    X, y = get_digits_data()
    loss_fn = nn.CrossEntropyLoss()
    baseline_losses = []
    anderson_losses = []

    for seed in seeds:
        torch.manual_seed(seed)
        model_b = SmallCNNTanh()
        model_aa = copy.deepcopy(model_b)

        loss_b, _ = train_baseline(model_b, X, y, loss_fn, steps=100, lr=0.1, momentum=0.7)
        loss_aa, _ = train_with_anderson(
            model_aa, X, y, loss_fn, steps=100, lr=0.1, momentum=0.7, window=5, aa_interval=10
        )

        baseline_losses.append(loss_b)
        anderson_losses.append(loss_aa)

    res = paired_analysis(baseline_losses, anderson_losses)
    print(format_results_table(res, title="SmallCNNTanh (Smooth): Baseline vs Anderson (60 seeds)"))
    write_experiment_result(
        os.path.join("results", "test_anderson_real_tanh.json"),
        experiment="SmallCNNTanh full-batch baseline vs safeguarded Anderson",
        config={"seeds": seeds, "steps": 100, "lr": 0.1, "momentum": 0.7, "window": 5, "aa_interval": 10},
        results=res,
        per_seed={"baseline_losses": baseline_losses, "anderson_losses": anderson_losses},
        notes=[
            "This preserves the historical lr=0.1 Tanh run; use the smoothness ablation script for same-seed same-lr causal checks."
        ],
    )
    return res


def main():
    print("=" * 68)
    print("  Real-Network Benchmarks & Activation Smoothness Isolation")
    res_mlp = test_tinymlp()
    res_relu = test_small_cnn_relu()
    res_tanh = test_small_cnn_tanh()
    print("=" * 68)
    return {"mlp": res_mlp, "relu": res_relu, "tanh": res_tanh}


if __name__ == "__main__":
    main()
