"""
Training execution loops and seed-matched paired comparison runners.

Implements:
1. Deterministic baseline SGD training
2. Safeguarded classical Type-II Anderson acceleration
3. Ito & Xue snapshot-residual Anderson acceleration
4. Seed-matched paired benchmarking across arbitrary seeds
"""

import copy

import numpy as np
import torch

from .anderson import (
    anderson_extrapolate_classical,
    anderson_extrapolate_ito_xue,
    get_full_state,
    get_position_only,
    passes_strict_descent_safeguard,
    set_full_state,
    set_position_only,
    state_is_sane,
)
from .statistics import paired_analysis

# ---------------------------------------------------------------------------
# Training Loops
# ---------------------------------------------------------------------------


def train_baseline(model, X, y, loss_fn, steps=100, lr=0.05, momentum=0.7, log_trajectory=False):
    """
    Standard full-batch momentum SGD training.

    Returns:
        final_loss (float)
        history (dict): containing 'losses' and optionally 'trajectory'
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    losses = []
    trajectory = []

    def eval_loss():
        with torch.no_grad():
            pred = model(X)
            return (loss_fn(pred, y) if y is not None else loss_fn(pred, X)).item()

    for _step in range(steps):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y) if y is not None else loss_fn(pred, X)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if log_trajectory:
            w = torch.cat([p.detach().flatten() for p in model.parameters()]).cpu().numpy()
            trajectory.append(w)

    final_loss = eval_loss() if steps > 0 else float("nan")
    return final_loss, {"losses": losses, "trajectory": trajectory}


def train_with_anderson(
    model,
    X,
    y,
    loss_fn,
    steps=100,
    lr=0.05,
    momentum=0.7,
    window=5,
    aa_interval=10,
    reg=1e-6,
    safeguard=True,
    log_trajectory=False,
):
    """
    Safeguarded Type-II Anderson Accelerated training (§5.10).

    Args:
        model: PyTorch model
        X: inputs
        y: targets (or None for autoencoders)
        loss_fn: criterion
        steps: total training steps
        lr: learning rate
        momentum: momentum coefficient
        window: Anderson history window m
        aa_interval: steps between acceleration attempts
        reg: Tikhonov regularization
        safeguard: if True, strictly requires loss(jump) < loss(pre-jump)
        log_trajectory: if True, log parameter positions

    Returns:
        final_loss (float)
        info (dict): containing losses, jump attempts, accepted jumps, etc.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    D = sum(p.numel() for p in model.parameters())

    def eval_loss():
        with torch.no_grad():
            pred = model(X)
            return (loss_fn(pred, y) if y is not None else loss_fn(pred, X)).item()

    state_history = [get_full_state(model, optimizer, D)]
    losses = []
    trajectory = []
    jumps_attempted = 0
    jumps_accepted = 0

    for step in range(steps):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y) if y is not None else loss_fn(pred, X)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        state_history.append(get_full_state(model, optimizer, D))

        if len(state_history) > window + 2:
            state_history.pop(0)

        # Attempt Anderson jump
        if step > 0 and step % aa_interval == 0 and len(state_history) >= window + 1:
            jumps_attempted += 1
            extrapolated = anderson_extrapolate_classical(state_history, window, reg=reg)

            if extrapolated is not None and state_is_sane(extrapolated):
                current_state = get_full_state(model, optimizer, D)
                loss_before_jump = eval_loss()

                set_full_state(model, optimizer, extrapolated, D)
                new_loss = eval_loss()

                if safeguard:
                    if not passes_strict_descent_safeguard(new_loss, loss_before_jump):
                        # Revert jump
                        set_full_state(model, optimizer, current_state, D)
                    else:
                        jumps_accepted += 1
                        # Reset history upon successful jump
                        state_history = [get_full_state(model, optimizer, D)]
                else:
                    jumps_accepted += 1
                    state_history = [get_full_state(model, optimizer, D)]

        if log_trajectory:
            w = torch.cat([p.detach().flatten() for p in model.parameters()]).cpu().numpy()
            trajectory.append(w)

    final_loss = eval_loss()
    return final_loss, {
        "losses": losses,
        "trajectory": trajectory,
        "jumps_attempted": jumps_attempted,
        "jumps_accepted": jumps_accepted,
    }


def train_with_anderson_ito_xue(
    model,
    X,
    y,
    loss_fn,
    steps=100,
    lr=0.05,
    momentum=0.7,
    window=5,
    aa_interval=10,
    reg=1e-6,
    safeguard=True,
    is_classification=True,
):
    """
    Ito & Xue (2025) Anderson-type prediction-residual snapshot combination.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    def eval_loss():
        with torch.no_grad():
            pred = model(X)
            return (loss_fn(pred, y) if y is not None else loss_fn(pred, X)).item()

    param_history = []
    pred_history = []
    losses = []
    jumps_attempted = 0
    jumps_accepted = 0

    # Target matrix for residual calculation
    if is_classification and y is not None:
        # Convert integer labels to one-hot for prediction-residual math
        num_classes = 10
        y_targets = np.eye(num_classes)[y.cpu().numpy()]
    elif y is not None:
        y_targets = y.cpu().numpy()
    else:
        y_targets = X.cpu().numpy()

    for step in range(steps):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y) if y is not None else loss_fn(pred, X)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        with torch.no_grad():
            curr_pred = model(X).cpu().numpy()
            curr_params = get_position_only(model)

        param_history.append(curr_params)
        pred_history.append(curr_pred)

        if len(param_history) > window:
            param_history.pop(0)
            pred_history.pop(0)

        if step > 0 and step % aa_interval == 0 and len(param_history) >= 2:
            jumps_attempted += 1
            x_new = anderson_extrapolate_ito_xue(param_history, pred_history, y_targets, reg=reg)

            if x_new is not None:
                current_params = get_position_only(model)
                loss_before = eval_loss()

                set_position_only(model, x_new)
                new_loss = eval_loss()

                if safeguard:
                    if not passes_strict_descent_safeguard(new_loss, loss_before):
                        set_position_only(model, current_params)
                    else:
                        jumps_accepted += 1
                        param_history = [get_position_only(model)]
                        pred_history = [model(X).detach().cpu().numpy()]
                else:
                    jumps_accepted += 1

    return eval_loss(), {
        "losses": losses,
        "jumps_attempted": jumps_attempted,
        "jumps_accepted": jumps_accepted,
    }


# ---------------------------------------------------------------------------
# Seed-Matched Paired Comparison Runner
# ---------------------------------------------------------------------------


def run_paired_comparison(
    model_fn,
    get_data_fn,
    loss_fn,
    seeds,
    steps=100,
    lr=0.05,
    momentum=0.7,
    window=5,
    aa_interval=10,
    reg=1e-6,
    method="classical",
    verbose=False,
):
    """
    Execute seed-matched paired benchmarking. For every seed, the model
    and optimizer are initialized identically for baseline and Anderson.
    """
    baseline_losses = []
    anderson_losses = []

    for seed in seeds:
        # Generate data if data depends on seed or fixed
        data = get_data_fn(seed)
        if len(data) == 2:
            X, y = data
        else:
            X, y = data[0], data[1]

        # Initialize baseline model
        torch.manual_seed(seed)
        np.random.seed(seed)
        model_base = model_fn()

        # Clone identical initial weights for Anderson
        model_aa = copy.deepcopy(model_base)

        # Run baseline
        loss_base, _ = train_baseline(model_base, X, y, loss_fn, steps=steps, lr=lr, momentum=momentum)

        # Run Anderson
        if method == "classical":
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
                reg=reg,
                safeguard=True,
            )
        elif method == "ito_xue":
            loss_aa, info = train_with_anderson_ito_xue(
                model_aa,
                X,
                y,
                loss_fn,
                steps=steps,
                lr=lr,
                momentum=momentum,
                window=window,
                aa_interval=aa_interval,
                reg=reg,
                safeguard=True,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        baseline_losses.append(loss_base)
        anderson_losses.append(loss_aa)

        if verbose:
            diff = loss_aa - loss_base
            print(f"Seed {seed:5d} | Baseline: {loss_base:.6f} | AA: {loss_aa:.6f} | Diff: {diff:+.6f}")

    return paired_analysis(baseline_losses, anderson_losses), baseline_losses, anderson_losses
