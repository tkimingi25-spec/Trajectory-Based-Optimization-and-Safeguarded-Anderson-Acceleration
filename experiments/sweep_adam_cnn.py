import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import get_digits_data
from src.models import SmallCNNTanh
from src.statistics import paired_analysis
from src.training import train_baseline, train_with_anderson


def sweep_adam_on_cnn():
    print("=" * 80)
    print("  ADVERSARIAL SWEEP: ADAM vs SAFEGUARDED ANDERSON ON SmallCNNTanh (Digits)")
    print("=" * 80)

    X, y = get_digits_data()
    loss_fn = nn.CrossEntropyLoss()
    seeds = list(range(8000, 8010))  # 10 seeds for fast check

    adam_lrs = [0.0005, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05]
    print("Sweeping Adam LRs on Digits (100 steps)...")

    adam_results = {}
    for lr in adam_lrs:
        losses = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = SmallCNNTanh()
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(100):
                opt.zero_grad()
                loss = loss_fn(model(X), y)
                loss.backward()
                opt.step()
            with torch.no_grad():
                losses.append(loss_fn(model(X), y).item())
        adam_results[lr] = losses
        print(f"  Adam lr={lr:<6} -> Mean CrossEntropy: {np.mean(losses):.6f} (Std: {np.std(losses):.6f})")

    best_adam_lr = min(adam_results.keys(), key=lambda lr: np.mean(adam_results[lr]))
    print(f"\n=> Best Adam LR: {best_adam_lr} (Mean: {np.mean(adam_results[best_adam_lr]):.6f})")

    # Anderson on base lr=0.1
    anderson_losses = []
    baseline_losses = []
    for seed in seeds:
        torch.manual_seed(seed)
        m1 = SmallCNNTanh()
        m2 = copy.deepcopy(m1)

        l_base, _ = train_baseline(m1, X, y, loss_fn, steps=100, lr=0.1, momentum=0.7)
        l_aa, _ = train_with_anderson(
            m2, X, y, loss_fn, steps=100, lr=0.1, momentum=0.7, window=5, aa_interval=10, safeguard=True
        )

        baseline_losses.append(l_base)
        anderson_losses.append(l_aa)

    print(f"  Baseline SGD-M (lr=0.1)        -> Mean CrossEntropy: {np.mean(baseline_losses):.6f}")
    print(f"  Safeguarded Anderson (lr=0.1)  -> Mean CrossEntropy: {np.mean(anderson_losses):.6f}")

    print("\nHead-to-Head Comparison on SmallCNNTanh (10 seeds):")
    print(
        f"{'Seed':<6} | {'Best Adam (' + str(best_adam_lr) + ')':<20} | {'Baseline SGD-M (0.1)':<22} | {'Safeguarded Anderson':<22}"
    )
    print("-" * 75)
    for i, seed in enumerate(seeds):
        print(
            f"{seed:<6} | {adam_results[best_adam_lr][i]:<20.6f} | {baseline_losses[i]:<22.6f} | {anderson_losses[i]:<22.6f}"
        )

    res = paired_analysis(adam_results[best_adam_lr], anderson_losses)
    print(f"\nAnderson vs Best Adam ({best_adam_lr}):")
    print(f"  * Anderson Wins: {res['wins']}/{res['n_seeds']}")
    print(f"  * Rel Reduction: {res['rel_reduction_pct']:+.2f}%")
    print(f"  * Paired t-test: p = {res['p_t']:.4e}")


if __name__ == "__main__":
    sweep_adam_on_cnn()
