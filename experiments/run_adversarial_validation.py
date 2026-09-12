import copy
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import TinyMLP
from src.data import get_tinymlp_data
from src.training import train_baseline, train_with_anderson
from src.statistics import paired_analysis

def run_adversarial_tuning_and_ablation():
    print("=" * 80)
    print("  ADVERSARIAL VALIDATION & HYPERPARAMETER GRID SWEEP (TinyMLP Regression)")
    print("=" * 80)

    loss_fn = nn.MSELoss()
    seeds = list(range(4000, 4030)) # 30 matched seeds
    
    # 1. Extensive LR sweep for Adam: finding best Adam tuning
    adam_lrs = [0.0005, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    print(f"\n[Step 1] Sweeping Adam learning rates over {adam_lrs} across 30 seeds...")
    
    adam_sweep_results = {}
    for lr in adam_lrs:
        losses = []
        for seed in seeds:
            X, y = get_tinymlp_data(N=200, seed=seed)
            torch.manual_seed(seed)
            model = TinyMLP()
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(150):
                opt.zero_grad()
                loss = loss_fn(model(X), y)
                loss.backward()
                opt.step()
            with torch.no_grad():
                losses.append(loss_fn(model(X), y).item())
        adam_sweep_results[lr] = losses
        print(f"  Adam lr={lr:<7} -> Mean MSE: {np.mean(losses):.6f} (Std: {np.std(losses):.6f})")

    # Find the best performing Adam lr
    best_adam_lr = min(adam_sweep_results.keys(), key=lambda lr: np.mean(adam_sweep_results[lr]))
    print(f"\n=> Best Tuned Adam Learning Rate: lr={best_adam_lr} (Mean MSE: {np.mean(adam_sweep_results[best_adam_lr]):.6f})")

    # 2. Extensive LR sweep for SGD-M
    sgdm_lrs = [0.01, 0.02, 0.05, 0.1, 0.2]
    print(f"\n[Step 2] Sweeping SGD-M learning rates over {sgdm_lrs} across 30 seeds...")
    sgdm_sweep_results = {}
    for lr in sgdm_lrs:
        losses = []
        for seed in seeds:
            X, y = get_tinymlp_data(N=200, seed=seed)
            torch.manual_seed(seed)
            model = TinyMLP()
            loss_val, _ = train_baseline(model, X, y, loss_fn, steps=150, lr=lr, momentum=0.7)
            losses.append(loss_val)
        sgdm_sweep_results[lr] = losses
        print(f"  SGD-M lr={lr:<6} -> Mean MSE: {np.mean(losses):.6f} (Std: {np.std(losses):.6f})")

    best_sgdm_lr = min(sgdm_sweep_results.keys(), key=lambda lr: np.mean(sgdm_sweep_results[lr]))
    print(f"\n=> Best Tuned SGD-M Learning Rate: lr={best_sgdm_lr} (Mean MSE: {np.mean(sgdm_sweep_results[best_sgdm_lr]):.6f})")

    # 3. Sweep Anderson (Ours) with best base SGD-M lr
    print(f"\n[Step 3] Running Safeguarded Anderson on base lr={best_sgdm_lr}...")
    anderson_losses = []
    for seed in seeds:
        X, y = get_tinymlp_data(N=200, seed=seed)
        torch.manual_seed(seed)
        model = TinyMLP()
        loss_val, _ = train_with_anderson(
            model, X, y, loss_fn, steps=150, lr=best_sgdm_lr, momentum=0.7, window=5, aa_interval=10, safeguard=True
        )
        anderson_losses.append(loss_val)
    print(f"  Safeguarded Anderson -> Mean MSE: {np.mean(anderson_losses):.6f} (Std: {np.std(anderson_losses):.6f})")

    # 4. Raw Per-Seed Results comparison
    print("\n" + "=" * 90)
    print("  RAW PER-SEED FINAL LOSS VALUES (30 SEEDS: 4000 to 4029)")
    print("=" * 90)
    print(f"{'Seed':<6} | {'Best Adam (' + str(best_adam_lr) + ')':<22} | {'Best SGD-M (' + str(best_sgdm_lr) + ')':<22} | {'Safeguarded Anderson':<22} | {'Delta vs Adam':<12}")
    print("-" * 90)
    
    adam_best_losses = adam_sweep_results[best_adam_lr]
    sgdm_best_losses = sgdm_sweep_results[best_sgdm_lr]

    for i, seed in enumerate(seeds):
        a_loss = adam_best_losses[i]
        s_loss = sgdm_best_losses[i]
        aa_loss = anderson_losses[i]
        delta = aa_loss - a_loss
        print(f"{seed:<6} | {a_loss:<22.6f} | {s_loss:<22.6f} | {aa_loss:<22.6f} | {delta:<+12.6f}")

    print("=" * 90)

    # 5. Statistical Head-to-Head Tests with Independently Tuned Optima
    print("\n" + "#" * 80)
    print("  HEAD-TO-HEAD STATISTICAL TESTS WITH INDEPENDENT OPTIMAL TUNING")
    print("#" * 80)

    # Anderson vs Best Adam
    res_adam = paired_analysis(adam_best_losses, anderson_losses)
    print(f"\nSafeguarded Anderson vs. Best Tuned Adam (lr={best_adam_lr}):")
    print(f"  * Anderson Wins:           {res_adam['wins']} / {res_adam['n_seeds']} ({res_adam['wins']/res_adam['n_seeds']*100:.1f}%)")
    print(f"  * Relative Loss Reduction: {res_adam['rel_reduction_pct']:+.2f}%")
    print(f"  * Paired t-test p-value:   {res_adam['p_t']:.4e}")
    print(f"  * Wilcoxon signed-rank p:  {res_adam['p_w']:.4e}")

    # Anderson vs Best SGD-M
    res_sgdm = paired_analysis(sgdm_best_losses, anderson_losses)
    print(f"\nSafeguarded Anderson vs. Best Tuned SGD-M (lr={best_sgdm_lr}):")
    print(f"  * Anderson Wins:           {res_sgdm['wins']} / {res_sgdm['n_seeds']} ({res_sgdm['wins']/res_sgdm['n_seeds']*100:.1f}%)")
    print(f"  * Relative Loss Reduction: {res_sgdm['rel_reduction_pct']:+.2f}%")
    print(f"  * Paired t-test p-value:   {res_sgdm['p_t']:.4e}")
    print(f"  * Wilcoxon signed-rank p:  {res_sgdm['p_w']:.4e}")

if __name__ == "__main__":
    run_adversarial_tuning_and_ablation()
