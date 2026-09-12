import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import get_tinymlp_data
from src.models import TinyMLP
from src.statistics import paired_analysis
from src.training import train_with_anderson


def run_tinymlp_benchmark():
    print("=" * 85, flush=True)
    print("  BENCHMARK: SAFEGUARDED ANDERSON vs INDUSTRY STANDARDS (TinyMLP Regression)", flush=True)
    print("=" * 85, flush=True)

    loss_fn = nn.MSELoss()
    seeds = list(range(4000, 4030))  # 30 paired seeds

    optimizers_to_test = [
        ("SGD-M (Standard Vision Baseline)", lambda m: torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.7)),
        (
            "NAG (Nesterov Accelerated Gradient)",
            lambda m: torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.7, nesterov=True),
        ),
        ("Adam (Industry Standard DL Optimizer)", lambda m: torch.optim.Adam(m.parameters(), lr=0.01)),
        ("AdamW (Decoupled Weight Decay)", lambda m: torch.optim.AdamW(m.parameters(), lr=0.01, weight_decay=1e-4)),
        ("RMSprop (Hinton Adaptive Gradient)", lambda m: torch.optim.RMSprop(m.parameters(), lr=0.01)),
    ]

    results = {name: [] for name, _ in optimizers_to_test}
    results["Safeguarded Anderson [w; v] (Ours)"] = []

    t0 = time.time()
    for seed in seeds:
        X, y = get_tinymlp_data(N=200, seed=seed)
        torch.manual_seed(seed)
        base_model = TinyMLP()

        # Standard optimizers
        for name, opt_fn in optimizers_to_test:
            m = copy.deepcopy(base_model)
            opt = opt_fn(m)
            for _ in range(150):
                opt.zero_grad()
                loss = loss_fn(m(X), y)
                loss.backward()
                opt.step()
            with torch.no_grad():
                results[name].append(loss_fn(m(X), y).item())

        # Safeguarded Anderson Acceleration
        m = copy.deepcopy(base_model)
        loss_aa, _ = train_with_anderson(
            m, X, y, loss_fn, steps=150, lr=0.05, momentum=0.7, window=5, aa_interval=10, safeguard=True
        )
        results["Safeguarded Anderson [w; v] (Ours)"].append(loss_aa)

    print(f"Completed 30 seeds across 6 optimizers in {time.time() - t0:.2f} seconds!\n", flush=True)

    print("=" * 85, flush=True)
    print(f"{'Method / Optimizer':<40} | {'Mean Final MSE':<18} | {'Std Dev':<12}", flush=True)
    print("=" * 85, flush=True)
    for name, losses in results.items():
        print(f"{name:<40} | {np.mean(losses):<18.6f} | {np.std(losses):<12.6f}", flush=True)
    print("=" * 85, flush=True)

    # Detailed Head-to-Head Stats
    anderson_losses = results["Safeguarded Anderson [w; v] (Ours)"]
    print("\n" + "#" * 85, flush=True)
    print("  HEAD-TO-HEAD STATISTICAL TESTS AGAINST SAFEGUARDED ANDERSON (30 SEEDS)", flush=True)
    print("#" * 85, flush=True)

    for name, losses in results.items():
        if name == "Safeguarded Anderson [w; v] (Ours)":
            continue
        analysis = paired_analysis(losses, anderson_losses)
        win_pct = (analysis["wins"] / analysis["n_seeds"]) * 100
        print(f"\nSafeguarded Anderson vs. {name}:")
        print(f"  * Anderson Win Rate:       {analysis['wins']} / {analysis['n_seeds']} ({win_pct:.1f}%)")
        print(f"  * Relative Loss Reduction: {analysis['rel_reduction_pct']:+.2f}%")
        print(
            f"  * Paired t-test p-value:   {analysis['p_t']:.4e} {'(Statistically Significant)' if analysis['p_t'] < 0.05 else '(Not Sig)'}"
        )
        print(f"  * Wilcoxon signed-rank p:  {analysis['p_w']:.4e}")


if __name__ == "__main__":
    run_tinymlp_benchmark()
