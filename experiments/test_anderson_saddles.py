"""
Anderson Acceleration on Baldi-Hornik Saddles (§5.10)

Demonstrates the first mechanism to generalize across both independently
constructed strict saddle landscapes:
1. Saddle #1 (D=24, d=6, k=2): 30 seeds
2. Saddle #2 (D=60, d=10, k=3): 30 seeds (where Mechanisms 4 & 5 failed)
3. Unsafeguarded failure check (demonstrating the necessity of strict descent)
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.landscapes import make_baldi_hornik_saddle
from src.training import train_baseline, train_with_anderson
from src.statistics import paired_analysis, format_results_table
from src.provenance import write_experiment_result


def run_saddle_benchmark(saddle_type, seeds, window=5, aa_interval=10, reg=1e-6, safeguard=True):
    baseline_losses = []
    anderson_losses = []

    for seed in seeds:
        model_b, X, loss_fn, meta = make_baldi_hornik_saddle(saddle_type=saddle_type)
        D = meta["D"]

        # Perturb slightly to break exact critical point symmetry
        rng = np.random.default_rng(seed)
        eps = rng.standard_normal(D) * 1e-4
        with torch.no_grad():
            idx = 0
            for p in model_b.parameters():
                n = p.numel()
                p.add_(torch.tensor(eps[idx:idx+n], dtype=p.dtype).view_as(p))
                idx += n

        model_aa = copy.deepcopy(model_b)

        # Baseline
        loss_b, _ = train_baseline(model_b, X, None, loss_fn, steps=150, lr=0.05, momentum=0.7)

        # Anderson
        loss_aa, _ = train_with_anderson(
            model_aa,
            X,
            None,
            loss_fn,
            steps=150,
            lr=0.05,
            momentum=0.7,
            window=window,
            aa_interval=aa_interval,
            reg=reg,
            safeguard=safeguard,
        )

        baseline_losses.append(loss_b)
        anderson_losses.append(loss_aa)

    return paired_analysis(baseline_losses, anderson_losses), baseline_losses, anderson_losses


def main():
    print("=" * 68)
    print("  Anderson Acceleration on Baldi-Hornik Strict Saddles (§5.10)")
    print("=" * 68)

    # 1. Saddle #1 (D=24, 30 seeds)
    print("\n[Running Saddle #1 Benchmark (30 seeds: 3000-3029)]...")
    res_s1, base_s1, aa_s1 = run_saddle_benchmark(saddle_type=1, seeds=range(3000, 3030), safeguard=True)
    print(format_results_table(res_s1, title="Saddle #1: Safeguarded Anderson Acceleration (D=24)"))
    write_experiment_result(
        os.path.join("results", "test_anderson_saddle1.json"),
        experiment="Baldi-Hornik Saddle #1 baseline vs safeguarded Anderson",
        config={"saddle_type": 1, "seeds": range(3000, 3030), "steps": 150, "lr": 0.05, "momentum": 0.7, "window": 5, "aa_interval": 10, "safeguard": True},
        results=res_s1,
        per_seed={"baseline_losses": base_s1, "anderson_losses": aa_s1},
    )

    # 2. Saddle #2 (D=60, 30 seeds)
    print("\n[Running Saddle #2 Benchmark (30 seeds: 3000-3029)]...")
    res_s2, base_s2, aa_s2 = run_saddle_benchmark(saddle_type=2, seeds=range(3000, 3030), safeguard=True)
    print(format_results_table(res_s2, title="Saddle #2: Safeguarded Anderson Acceleration (D=60)"))
    write_experiment_result(
        os.path.join("results", "test_anderson_saddle2.json"),
        experiment="Baldi-Hornik Saddle #2 baseline vs safeguarded Anderson",
        config={"saddle_type": 2, "seeds": range(3000, 3030), "steps": 150, "lr": 0.05, "momentum": 0.7, "window": 5, "aa_interval": 10, "safeguard": True},
        results=res_s2,
        per_seed={"baseline_losses": base_s2, "anderson_losses": aa_s2},
    )

    # 3. Unsafeguarded Failure Demonstration (§5.10)
    print("\n[Demonstrating Unsafeguarded Anderson Failure on Saddle #1 (5 seeds)]...")
    res_unsafeguarded, base_unsafeguarded, aa_unsafeguarded = run_saddle_benchmark(
        saddle_type=1, seeds=range(3000, 3005), safeguard=False
    )
    print(f"Unsafeguarded Anderson Mean Loss: {res_unsafeguarded['mean_test']:.4f} vs Baseline: {res_unsafeguarded['mean_baseline']:.4f}")
    print(f"Wins: {res_unsafeguarded['wins']}/5 (Extrapolations without safeguard blow up near saddle noise)")
    write_experiment_result(
        os.path.join("results", "test_anderson_saddle1_unsafeguarded.json"),
        experiment="Baldi-Hornik Saddle #1 baseline vs unsafeguarded Anderson",
        config={"saddle_type": 1, "seeds": range(3000, 3005), "steps": 150, "lr": 0.05, "momentum": 0.7, "window": 5, "aa_interval": 10, "safeguard": False},
        results=res_unsafeguarded,
        per_seed={"baseline_losses": base_unsafeguarded, "anderson_losses": aa_unsafeguarded},
    )
    print("=" * 68)


if __name__ == "__main__":
    main()
