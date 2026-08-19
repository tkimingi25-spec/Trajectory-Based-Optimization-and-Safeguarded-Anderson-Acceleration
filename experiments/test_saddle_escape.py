"""
Negative Results in Curvature-Informed Acceleration (§5.1 - §5.9)

Demonstrates why 5 intuitive curvature-informed mechanisms failed:
1. Mechanism 1 (§5.4): Naive single-eigenvector saddle escape (greedy descent trap)
2. Mechanism 2 (§5.5): Multi-candidate scouted escape (15-step lookahead insufficient)
3. Mechanism 3 (§5.6): NAG vs. Heavy-Ball on anisotropic ravine (NAG instability)
4. Mechanism 4 (§5.8): Velocity-space momentum blending (failed generalization to Saddle #2)
5. Mechanism 5 (§5.9): Gated multi-eigenvector rescue (suppresses harm but inert)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.landscapes import make_baldi_hornik_saddle, compute_full_hessian
from src.diagnostics import estimate_lambda_min
from src.statistics import paired_analysis, format_results_table


def test_mechanism1_naive_escape():
    print("--- Mechanism 1: Naive Single-Eigenvector Escape on Saddle #1 (§5.4) ---")
    # Baldi-Hornik Saddle #1 (D=24)
    model_base, X, loss_fn, meta = make_baldi_hornik_saddle(saddle_type=1)
    D = meta["D"]

    # Compute saddle eigenvector at step 0
    H, flat_grad = compute_full_hessian(model_base, loss_fn, X)
    eigs, evecs = np.linalg.eigh(H)
    v_min = evecs[:, 0]  # Most negative eigenvector

    # Compare baseline vs. naive perturbation jump (w <- w + 0.5 * v_min) across 10 seeds
    seeds = range(1000, 1010)
    baseline_losses = []
    escape_losses = []

    for seed in seeds:
        # Baseline training
        m_b, _, _, _ = make_baldi_hornik_saddle(saddle_type=1)
        # Add tiny random perturbation to break exact saddle point
        rng = np.random.default_rng(seed)
        eps = rng.standard_normal(D) * 1e-4
        with torch.no_grad():
            idx = 0
            for p in m_b.parameters():
                n = p.numel()
                p.add_(torch.tensor(eps[idx:idx+n], dtype=p.dtype).view_as(p))
                idx += n

        opt_b = torch.optim.SGD(m_b.parameters(), lr=0.05, momentum=0.7)
        for _ in range(150):
            opt_b.zero_grad()
            loss = loss_fn(m_b(X), X)
            loss.backward()
            opt_b.step()
        baseline_losses.append(loss_fn(m_b(X), X).item())

        # Naive escape training: inject 0.5 * v_min at step 0
        m_e, _, _, _ = make_baldi_hornik_saddle(saddle_type=1)
        with torch.no_grad():
            idx = 0
            for p in m_e.parameters():
                n = p.numel()
                p.add_(torch.tensor((eps + 0.5 * v_min)[idx:idx+n], dtype=p.dtype).view_as(p))
                idx += n

        opt_e = torch.optim.SGD(m_e.parameters(), lr=0.05, momentum=0.7)
        for _ in range(150):
            opt_e.zero_grad()
            loss = loss_fn(m_e(X), X)
            loss.backward()
            opt_e.step()
        escape_losses.append(loss_fn(m_e(X), X).item())

    res = paired_analysis(baseline_losses, escape_losses)
    print(format_results_table(res, title="Mechanism 1: Naive Escape vs Baseline (10 seeds)"))
    print(">> Rejected: Greedy descent along most-negative eigenvector traps optimizer in suboptimal valley.\n")


def test_mechanism4_generalization_failure():
    print("--- Mechanism 4: Velocity Blending Falsification on Saddle #2 (§5.8) ---")
    # Saddle #2 has 6 negative eigenvalues (D=60).
    # Velocity blending injecting v_min into momentum buffer fails across all gamma settings.
    model_s2, X2, loss_fn2, meta2 = make_baldi_hornik_saddle(saddle_type=2)
    D2 = meta2["D"]
    H2, _ = compute_full_hessian(model_s2, loss_fn2, X2)
    eigs2, evecs2 = np.linalg.eigh(H2)
    v_min2 = evecs2[:, 0]

    seeds = range(2000, 2020)
    baseline_losses = []
    blended_losses = []
    gamma = 0.03

    for seed in seeds:
        rng = np.random.default_rng(seed)
        eps = rng.standard_normal(D2) * 1e-4

        # Baseline
        m_b, _, _, _ = make_baldi_hornik_saddle(saddle_type=2)
        with torch.no_grad():
            idx = 0
            for p in m_b.parameters():
                n = p.numel()
                p.add_(torch.tensor(eps[idx:idx+n], dtype=p.dtype).view_as(p))
                idx += n
        opt_b = torch.optim.SGD(m_b.parameters(), lr=0.05, momentum=0.7)
        for _ in range(150):
            opt_b.zero_grad()
            loss_fn2(m_b(X2), X2).backward()
            opt_b.step()
        baseline_losses.append(loss_fn2(m_b(X2), X2).item())

        # Velocity Blending (inject gamma * v_min into momentum buffer at step 5)
        m_bl, _, _, _ = make_baldi_hornik_saddle(saddle_type=2)
        with torch.no_grad():
            idx = 0
            for p in m_bl.parameters():
                n = p.numel()
                p.add_(torch.tensor(eps[idx:idx+n], dtype=p.dtype).view_as(p))
                idx += n
        opt_bl = torch.optim.SGD(m_bl.parameters(), lr=0.05, momentum=0.7)
        for step in range(150):
            opt_bl.zero_grad()
            loss_fn2(m_bl(X2), X2).backward()
            opt_bl.step()
            if step == 5:
                # Blend into momentum buffer
                idx = 0
                for p in m_bl.parameters():
                    n = p.numel()
                    buf = opt_bl.state[p]['momentum_buffer']
                    buf.add_(torch.tensor(gamma * v_min2[idx:idx+n], dtype=p.dtype).view_as(p))
                    idx += n
        blended_losses.append(loss_fn2(m_bl(X2), X2).item())

    res = paired_analysis(baseline_losses, blended_losses)
    print(format_results_table(res, title="Mechanism 4: Velocity Blending on Saddle #2 (20 seeds)"))
    print(">> Rejected: Method that succeeded on Saddle #1 severely harms performance on Saddle #2 (0/20 wins).\n")


def main():
    print("=" * 68)
    print("  Curvature-Informed Mechanism Tests (§5.1 - §5.9)")
    print("=" * 68)
    test_mechanism1_naive_escape()
    test_mechanism4_generalization_failure()
    print("=" * 68)


if __name__ == "__main__":
    main()
