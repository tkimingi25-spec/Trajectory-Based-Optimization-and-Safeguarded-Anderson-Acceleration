"""
Anderson acceleration implementations.

Two structurally different formulations:
1. Classical Type-II (Walker & Ni 2011): extrapolates the fixed-point residual
   g(x) - x from the optimizer's own position+velocity state history.
2. Ito & Xue (2025): minimizes the weighted sum of prediction residuals
   (psi(x_i) - y) across parameter snapshots, subject to weights summing to 1.

Both require a strict-descent safeguard under non-convex optimization.
Without it, both catastrophically fail (documented in §5.10).
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# State management: position + velocity concatenation
# ---------------------------------------------------------------------------

def get_full_state(model, optimizer, D):
    """
    Extract the fixed-point state = (position, velocity) concatenated.

    Extrapolating both together avoids a stale-momentum-buffer inconsistency
    that was identified in §5.7: if only position is extrapolated, the
    momentum buffer retains stale velocity from the pre-jump trajectory.
    A control test showed zeroing vs. leaving stale made no material
    difference — but extrapolating both is cleaner and avoids the issue
    entirely.

    Args:
        model: PyTorch model
        optimizer: SGD optimizer with momentum
        D: number of parameters (position dimension; full state is 2*D)

    Returns:
        numpy array of shape (2*D,) — [position; velocity]
    """
    w = torch.cat([p.detach().flatten() for p in model.parameters()])
    v_parts = []
    for p in model.parameters():
        if p in optimizer.state and 'momentum_buffer' in optimizer.state[p]:
            v_parts.append(optimizer.state[p]['momentum_buffer'].detach().flatten())
        else:
            v_parts.append(torch.zeros_like(p).flatten())
    v = torch.cat(v_parts)
    return torch.cat([w, v]).cpu().numpy()


def set_full_state(model, optimizer, state_vec, D):
    """
    Restore a full state vector (position + velocity) into model and optimizer.

    Args:
        model: PyTorch model
        optimizer: SGD optimizer with momentum
        state_vec: numpy array of shape (2*D,) — [position; velocity]
        D: number of parameters
    """
    w, v = state_vec[:D], state_vec[D:]
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(torch.tensor(w[idx:idx + n], dtype=p.dtype).view_as(p))
        if p not in optimizer.state:
            optimizer.state[p] = {}
        optimizer.state[p]['momentum_buffer'] = (
            torch.tensor(v[idx:idx + n], dtype=p.dtype).view_as(p)
        )
        idx += n


def get_position_only(model):
    """Extract only the position (parameters) as a numpy vector."""
    return torch.cat([p.detach().flatten() for p in model.parameters()]).cpu().numpy()


def set_position_only(model, w_vec):
    """Set only the position (parameters) from a numpy vector."""
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(torch.tensor(w_vec[idx:idx + n], dtype=p.dtype).view_as(p))
        idx += n


# ---------------------------------------------------------------------------
# Classical Type-II Anderson acceleration (Walker & Ni 2011)
# ---------------------------------------------------------------------------

def anderson_extrapolate_classical(states, window, reg=1e-6):
    """
    Type-II Anderson acceleration with Tikhonov regularization.

    Given the last (window+1) states, computes the least-squares optimal
    linear combination to extrapolate the next iterate.

    CRITICAL NOTE: Regularization ALONE is not sufficient to prevent blow-up
    near slow trajectories. The strict-descent safeguard in the calling loop
    is essential — see §5.10 for the documented failure without it.

    Args:
        states: list of numpy arrays, each a full state vector
        window: Anderson window size m (uses last m+1 states)
        reg: Tikhonov regularization parameter

    Returns:
        numpy array (extrapolated state) or None if insufficient history
    """
    recent = states[-(window + 1):]
    if len(recent) < window + 1:
        return None

    X_hist = np.array(recent[:-1])  # states 0..m-1
    G_hist = np.array(recent[1:])   # states 1..m (the "g(x)" iterates)

    # Fixed-point residuals: f_i = g(x_i) - x_i
    F = G_hist - X_hist
    f_m = F[-1]

    if len(F) < 2:
        return None

    # Differences from the most recent residual
    F_diff = (f_m - F[:-1]).T  # shape: (state_dim, m-1)

    if F_diff.shape[1] == 0:
        return None

    # Regularized least squares: minimize ||f_m - F_diff @ gamma||^2 + reg * ||gamma||^2
    A = F_diff.T @ F_diff + reg * np.eye(F_diff.shape[1])
    b = F_diff.T @ f_m

    try:
        gamma = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    # Combination weights: alpha_i = gamma_i (i < m), alpha_m = 1 - sum(gamma)
    alpha = np.append(gamma, 1.0 - gamma.sum())

    # Extrapolated state: weighted combination of the "g" iterates
    return G_hist.T @ alpha


# ---------------------------------------------------------------------------
# Ito & Xue formulation (prediction-residual snapshot combination)
# ---------------------------------------------------------------------------

def anderson_extrapolate_ito_xue(param_snapshots, predictions, targets, reg=1e-6):
    """
    Ito & Xue (2025) Anderson-type acceleration.

    Structurally different from classical Type-II: the residual is the
    prediction error (psi(x_i) - y), and the method finds the least-squares
    combination of parameter snapshots that minimizes the total weighted
    prediction residual, subject to weights summing to 1.

    This is closer to a residual-weighted model-snapshot combination —
    in spirit closer to Stochastic Weight Averaging (Izmailov et al., 2018)
    blended with Anderson mixing.

    Args:
        param_snapshots: list of numpy arrays, each a parameter vector
        predictions: list of numpy arrays, each the model's prediction output
        targets: numpy array, the true labels/targets
        reg: Tikhonov regularization parameter

    Returns:
        numpy array (combined parameter vector) or None
    """
    m = len(param_snapshots)
    if m < 2:
        return None

    # Prediction residuals: r_i = psi(x_i) - y
    residuals = [pred - targets for pred in predictions]
    R = np.column_stack([r.flatten() for r in residuals])  # shape: (N*output_dim, m)

    # Minimize ||R @ alpha||^2 subject to sum(alpha) = 1
    # using the KKT system:
    #   [G  1][alpha] = [0]
    #   [1' 0][lambda]  [1]
    G = R.T @ R + reg * np.eye(m)  # Gram matrix with regularization
    ones = np.ones(m)

    try:
        kkt = np.zeros((m + 1, m + 1), dtype=np.float64)
        kkt[:m, :m] = G
        kkt[:m, m] = ones
        kkt[m, :m] = ones
        rhs = np.zeros(m + 1, dtype=np.float64)
        rhs[m] = 1.0
        solution = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError:
        return None

    alpha = solution[:m]

    # Combined parameters
    X = np.column_stack(param_snapshots)
    x_new = X @ alpha

    if not np.all(np.isfinite(x_new)):
        return None

    return x_new


# ---------------------------------------------------------------------------
# Safeguard utilities
# ---------------------------------------------------------------------------

def passes_strict_descent_safeguard(new_loss, old_loss):
    """
    Strict-descent safeguard — the critical fix documented in §5.10.

    Accept the extrapolated jump ONLY if it strictly lowers the loss.
    A loose tolerance (e.g., "accept if not >3x worse") was tested and
    found insufficient; only zero-tolerance strict descent prevented
    the numerical blow-up.

    Args:
        new_loss: loss after the proposed jump
        old_loss: loss before the jump

    Returns:
        True if the jump should be accepted
    """
    if not np.isfinite(new_loss):
        return False
    return new_loss < old_loss


def state_is_sane(state_vec, max_abs=1e4):
    """Check that extrapolated state doesn't contain extreme values."""
    return bool(np.all(np.isfinite(state_vec)) and np.max(np.abs(state_vec)) < max_abs)
