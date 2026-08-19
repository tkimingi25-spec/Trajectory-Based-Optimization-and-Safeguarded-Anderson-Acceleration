"""
Diagnostic telemetry and pathology detection tools.

Implements:
1. Trajectory PCA (Engine A, §4.3)
2. Pearlmutter Hessian-vector products (double-backward autograd and finite-difference)
3. Shifted power iteration for matrix-free minimum eigenvalue estimation (§4.3, §5.3)
4. Strict saddle detection (paired zero-gradient + negative curvature gating)
5. Step explosion guard (§4.4)
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Trajectory PCA (§4.3)
# ---------------------------------------------------------------------------

def trajectory_pca(trajectory_matrix):
    """
    Perform PCA on the optimization trajectory matrix W in R^(T x D).

    Args:
        trajectory_matrix: np.ndarray of shape (T, D) containing parameter checkpoints

    Returns:
        explained_variance_ratio: np.ndarray of shape (min(T, D),)
        components: principal component vectors
        projections: trajectory projected onto principal components
    """
    W = np.asarray(trajectory_matrix)
    # Center the trajectory across time steps
    W_centered = W - W.mean(axis=0, keepdims=True)

    U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)
    explained_variance = (S ** 2) / (len(W) - 1)
    total_var = explained_variance.sum()
    explained_variance_ratio = (
        explained_variance / total_var if total_var > 0 else np.zeros_like(explained_variance)
    )

    projections = U * S
    return explained_variance_ratio, Vt, projections


# ---------------------------------------------------------------------------
# Hessian-Vector Products (§4.3)
# ---------------------------------------------------------------------------

def hessian_vector_product_autograd(model, loss_fn, X, y, v):
    """
    Exact Hessian-vector product H @ v via PyTorch double-backward autograd.

    Args:
        model: PyTorch model
        loss_fn: loss function
        X: inputs
        y: targets (or None for autoencoders)
        v: numpy array or torch tensor of length D

    Returns:
        hv: numpy array of length D representing H @ v
    """
    if isinstance(v, np.ndarray):
        v_t = torch.tensor(v, dtype=torch.float32)
    else:
        v_t = v

    params = list(model.parameters())
    pred = model(X)
    loss = loss_fn(pred, y) if y is not None else loss_fn(pred, X)

    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.flatten() for g in grads])

    # Inner product with v
    grad_v = (flat_grad * v_t).sum()

    # Second backward gives H @ v
    hv_grads = torch.autograd.grad(grad_v, params, retain_graph=False)
    flat_hv = torch.cat([g.flatten() for g in hv_grads])
    return flat_hv.detach().cpu().numpy()


def pearlmutter_hv_fd(grad_fn, w, v, r=1e-5):
    """
    Centered finite-difference Hessian-vector product:
    Hv approx (grad(w + r*v) - grad(w - r*v)) / (2*r)
    """
    return (grad_fn(w + r * v) - grad_fn(w - r * v)) / (2 * r)


# ---------------------------------------------------------------------------
# Shifted Power Iteration for Minimum Eigenvalue (§4.3)
# ---------------------------------------------------------------------------

def _power_iterate(apply_op, D, iters=300, seed=1):
    """Run power iteration on a linear operator."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(D)
    v /= np.linalg.norm(v) + 1e-12

    for _ in range(iters):
        av = apply_op(v)
        n = np.linalg.norm(av)
        if n < 1e-12:
            break
        v = av / n
    return v


def estimate_lambda_min(apply_hv, D, iters=300, seed=1):
    """
    Two-stage shifted power iteration to find the true minimum eigenvalue (lambda_min)
    and its eigenvector.

    BUG HISTORY (§4.3): Plain power iteration on -H converges to the eigenvalue
    of LARGEST ABSOLUTE MAGNITUDE, which is lambda_max if |lambda_max| > |lambda_min|.
    Fix: Two-stage shifted iteration.
    Stage 1: Find lambda_max via power iteration on H.
    Stage 2: Shift the spectrum: (shift * I - H) has dominant eigenvalue (shift - lambda_min),
             which is strictly positive and maximal for shift = |lambda_max| + 5.0.

    Args:
        apply_hv: function taking v -> H @ v (np.ndarray)
        D: dimension
        iters: max power iterations
        seed: RNG seed

    Returns:
        lambda_min_est: estimated minimum eigenvalue (float)
        v_min: estimated minimum eigenvector (np.ndarray of length D)
    """
    # Stage 1: Find lambda_max
    v_max = _power_iterate(apply_hv, D, iters=iters, seed=seed)
    hv_max = apply_hv(v_max)
    lambda_max_est = float(v_max @ hv_max)

    # Stage 2: Shifted power iteration
    shift = abs(lambda_max_est) + 5.0
    v_min = _power_iterate(
        lambda v: shift * v - apply_hv(v),
        D,
        iters=iters,
        seed=seed + 1,
    )
    hv_min = apply_hv(v_min)
    lambda_min_est = float(v_min @ hv_min)

    return lambda_min_est, v_min


def saddle_check(grad_norm, apply_hv, D, grad_norm_threshold=0.05, iters=300, seed=1):
    """
    Paired strict saddle detector (§4.3, §5.3):
    Checks both near-zero gradient AND negative curvature.
    """
    if grad_norm > grad_norm_threshold:
        return {
            "stalled": False,
            "saddle_flag": False,
            "grad_norm": grad_norm,
            "lambda_min": None,
        }

    lambda_min, v_min = estimate_lambda_min(apply_hv, D, iters=iters, seed=seed)
    return {
        "stalled": True,
        "saddle_flag": bool(lambda_min < -1e-4),
        "grad_norm": grad_norm,
        "lambda_min": lambda_min,
        "v_min": v_min,
    }


# ---------------------------------------------------------------------------
# Step Explosion Guard (§4.4)
# ---------------------------------------------------------------------------

class ExplosionGuard:
    """
    Monitors parameter update step sizes (||Delta w_t||) to catch numerical
    divergence before loss explodes to infinity/NaN.
    """
    def __init__(self, max_step_norm=3.0):
        self.max_step_norm = max_step_norm
        self.triggered = False
        self.trigger_step = None
        self.trigger_delta = None

    def check(self, step, prev_params, curr_params):
        delta = np.linalg.norm(curr_params - prev_params)
        if delta > self.max_step_norm:
            self.triggered = True
            self.trigger_step = step
            self.trigger_delta = delta
            return False  # Failed guard
        return True  # Passed guard
