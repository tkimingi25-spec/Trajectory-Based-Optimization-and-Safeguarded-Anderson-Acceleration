"""
Optimization landscape constructions and analytical models.

Implements:
1. Anisotropic quadratic ravines (Tier 1 pedagogical track, §3)
2. Baldi-Hornik strict saddle constructions for LinearAutoencoder (Tier 2 & 3, §5.3)
3. Exact full-Hessian computation for low-dimensional models
"""

import numpy as np
import torch
import torch.nn as nn

from .data import get_baldi_hornik_data
from .models import LinearAutoencoder

# ---------------------------------------------------------------------------
# Tier 1: Quadratic Ravine Optimization (§3.4)
# ---------------------------------------------------------------------------


def run_quadratic_optimization(steps=400, lr=0.0013, momentum=0.7, a=20.0, b=1.0, seed=42):
    """
    Run momentum gradient descent on an anisotropic quadratic ravine:
    L(w1, w2) = 0.5 * (a * w1^2 + b * w2^2)

    Tuned for monotonic descent (no oscillation) at lr=0.0013 < lr_critical=0.001334.

    Returns:
        trajectory: list of dicts with keys (step, w1, w2, v1, v2, loss)
        a, b: quadratic coefficients
    """
    rng = np.random.default_rng(seed)
    w1 = float(rng.uniform(6.0, 10.0))
    w2 = float(rng.uniform(6.0, 10.0))
    v1 = 0.0
    v2 = 0.0
    trajectory = []

    for step in range(steps):
        loss = 0.5 * (a * (w1**2) + b * (w2**2))
        grad_w1 = a * w1
        grad_w2 = b * w2
        record = {
            "step": step,
            "w1": round(w1, 6),
            "w2": round(w2, 6),
            "v1": round(v1, 6),
            "v2": round(v2, 6),
            "loss": round(loss, 6),
        }
        trajectory.append(record)
        v1 = momentum * v1 - lr * grad_w1
        v2 = momentum * v2 - lr * grad_w2
        w1 += v1
        w2 += v2

    return trajectory, a, b


def verify_monotonic_trajectory(trajectory):
    """
    Verify that a 2D trajectory is strictly monotonic in |w1|, |w2|, and loss.
    """
    w1_increases = sum(
        1 for i in range(1, len(trajectory)) if abs(trajectory[i]["w1"]) > abs(trajectory[i - 1]["w1"]) + 1e-9
    )
    w2_increases = sum(
        1 for i in range(1, len(trajectory)) if abs(trajectory[i]["w2"]) > abs(trajectory[i - 1]["w2"]) + 1e-9
    )
    loss_increases = sum(
        1 for i in range(1, len(trajectory)) if trajectory[i]["loss"] > trajectory[i - 1]["loss"] + 1e-9
    )
    return w1_increases, w2_increases, loss_increases


# ---------------------------------------------------------------------------
# Tier 2 & 3: Baldi-Hornik Strict Saddle Models (§5.3)
# ---------------------------------------------------------------------------


def make_baldi_hornik_saddle(saddle_type=1):
    """
    Construct a LinearAutoencoder initialized at a strict Baldi-Hornik saddle point.

    Saddle #1: d=6, k=2, projection onto directions {2, 3} (true optimum is {0, 1})
    Saddle #2: d=10, k=3, projection onto directions {0, 2, 4} (true optimum is {0, 1, 2})

    Returns:
        model: LinearAutoencoder instance
        X: data tensor
        loss_fn: MSELoss instance
        metadata: dict with theoretical properties
    """
    X, d, k, saddle_idx, opt_idx, variances = get_baldi_hornik_data(saddle_type)
    model = LinearAutoencoder(d, k)

    with torch.no_grad():
        W = torch.zeros(k, d)
        for i, idx in enumerate(saddle_idx):
            W[i, idx] = 1.0
        model.enc.weight.copy_(W)
        model.dec.weight.copy_(W.T)

    loss_fn = nn.MSELoss()

    # Calculate analytical global optimum and saddle losses
    # Autoencoder MSE loss on diagonal covariance: sum_{i not in selected} var_i / d (or sum)
    # With MSE across all N samples:
    with torch.no_grad():
        saddle_loss = loss_fn(model(X), X).item()

    metadata = {
        "d": d,
        "k": k,
        "saddle_idx": saddle_idx,
        "opt_idx": opt_idx,
        "saddle_loss": saddle_loss,
        "D": sum(p.numel() for p in model.parameters()),
    }
    return model, X, loss_fn, metadata


def compute_full_hessian(model, loss_fn, X, y=None):
    """
    Compute the exact D x D Hessian matrix via PyTorch autograd double-backward.

    Args:
        model: PyTorch model
        loss_fn: loss function
        X: inputs
        y: targets (if None, autoencoder reconstruction loss_fn(pred, X) is used)

    Returns:
        H (np.ndarray): D x D Hessian matrix
        flat_grad (torch.Tensor): length-D gradient vector
    """
    params = list(model.parameters())
    pred = model(X)
    loss = loss_fn(pred, y) if y is not None else loss_fn(pred, X)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.flatten() for g in grads])
    D = sum(p.numel() for p in params)

    H = torch.zeros(D, D)
    for i in range(D):
        gi = torch.autograd.grad(flat_grad[i], params, retain_graph=True)
        H[i] = torch.cat([g.flatten() for g in gi])

    return H.detach().cpu().numpy(), flat_grad.detach()
