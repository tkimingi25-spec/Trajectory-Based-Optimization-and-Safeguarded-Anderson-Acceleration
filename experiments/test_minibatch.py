"""
Mini-Batch SGD Evaluation (Test #2)

Tests Anderson-style acceleration under stochastic mini-batch gradients with
paired initialization and identical per-seed batch schedules.

Variants:
1. Naive classical Anderson, safeguarded on the current mini-batch.
2. Classical Anderson, safeguarded on the full dataset.
3. Sparse-history classical Anderson, safeguarded on the full dataset.
4. Ito/Xue prediction-residual snapshot combination, safeguarded on the full dataset.
"""

import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.anderson import (
    anderson_extrapolate_classical,
    anderson_extrapolate_ito_xue,
    get_full_state,
    get_position_only,
    passes_strict_descent_safeguard,
    set_full_state,
    set_position_only,
    state_is_sane,
)
from src.data import get_digits_data
from src.models import SmallCNNTanh
from src.provenance import write_experiment_result
from src.statistics import format_results_table, paired_analysis


def make_batch_schedule(n_samples, batch_size, epochs, seed):
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(epochs):
        permutation = torch.randperm(n_samples, generator=generator)
        for start in range(0, n_samples, batch_size):
            batches.append(permutation[start:start + batch_size])
    return batches


def eval_loss(model, X, y, loss_fn):
    with torch.no_grad():
        return loss_fn(model(X), y).item()


def train_minibatch_baseline(model, X, y, batches, loss_fn, lr=0.05, momentum=0.7):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for batch_idx in batches:
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
    return eval_loss(model, X, y, loss_fn)


def train_minibatch_anderson_classical(
    model,
    X,
    y,
    batches,
    loss_fn,
    lr=0.05,
    momentum=0.7,
    window=5,
    aa_interval=5,
    history_stride=1,
    safeguard_dataset="full",
):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    D = sum(p.numel() for p in model.parameters())
    state_history = [get_full_state(model, optimizer, D)]
    jumps_attempted = 0
    jumps_accepted = 0

    for step, batch_idx in enumerate(batches, start=1):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

        if step % history_stride == 0:
            state_history.append(get_full_state(model, optimizer, D))
            if len(state_history) > window + 2:
                state_history.pop(0)

        if step % aa_interval == 0 and len(state_history) >= window + 1:
            jumps_attempted += 1
            extrapolated = anderson_extrapolate_classical(state_history, window)
            if extrapolated is None or not state_is_sane(extrapolated):
                continue

            current_state = get_full_state(model, optimizer, D)
            if safeguard_dataset == "batch":
                loss_before = eval_loss(model, X_batch, y_batch, loss_fn)
            elif safeguard_dataset == "full":
                loss_before = eval_loss(model, X, y, loss_fn)
            else:
                raise ValueError(f"Unknown safeguard dataset: {safeguard_dataset}")

            set_full_state(model, optimizer, extrapolated, D)
            if safeguard_dataset == "batch":
                loss_after = eval_loss(model, X_batch, y_batch, loss_fn)
            else:
                loss_after = eval_loss(model, X, y, loss_fn)

            if passes_strict_descent_safeguard(loss_after, loss_before):
                jumps_accepted += 1
                state_history = [get_full_state(model, optimizer, D)]
            else:
                set_full_state(model, optimizer, current_state, D)

    return eval_loss(model, X, y, loss_fn), {
        "jumps_attempted": jumps_attempted,
        "jumps_accepted": jumps_accepted,
    }


def train_minibatch_ito_xue(
    model,
    X,
    y,
    batches,
    loss_fn,
    lr=0.05,
    momentum=0.7,
    window=5,
    aa_interval=5,
    snapshot_stride=5,
    reg=1e-6,
    safeguard=True,
):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    param_history = []
    pred_history = []
    jumps_attempted = 0
    jumps_accepted = 0
    targets = np.eye(10)[y.cpu().numpy()]

    for step, batch_idx in enumerate(batches, start=1):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

        if step % snapshot_stride == 0:
            with torch.no_grad():
                param_history.append(get_position_only(model))
                pred_history.append(model(X).cpu().numpy())
            if len(param_history) > window:
                param_history.pop(0)
                pred_history.pop(0)

        if step % aa_interval == 0 and len(param_history) >= 2:
            jumps_attempted += 1
            x_new = anderson_extrapolate_ito_xue(param_history, pred_history, targets, reg=reg)
            if x_new is None or not np.all(np.isfinite(x_new)):
                continue

            current_params = get_position_only(model)
            loss_before = eval_loss(model, X, y, loss_fn)
            set_position_only(model, x_new)
            loss_after = eval_loss(model, X, y, loss_fn)

            if not safeguard or passes_strict_descent_safeguard(loss_after, loss_before):
                jumps_accepted += 1
                param_history = [get_position_only(model)]
                pred_history = [model(X).detach().cpu().numpy()]
            else:
                set_position_only(model, current_params)

    return eval_loss(model, X, y, loss_fn), {
        "jumps_attempted": jumps_attempted,
        "jumps_accepted": jumps_accepted,
    }


def main(n_seeds=60, seed_start=9000, epochs=15, batch_size=64, output_path=None):
    print("=" * 68)
    print("  Mini-Batch SGD Evaluation (Test #2)")
    print("=" * 68)

    X, y = get_digits_data()
    loss_fn = nn.CrossEntropyLoss()
    seeds = range(seed_start, seed_start + n_seeds)

    variant_losses = {
        "baseline": [],
        "classical_batch_safeguard": [],
        "classical_full_safeguard": [],
        "classical_sparse_full_safeguard": [],
        "ito_xue_full_safeguard": [],
    }
    variant_infos = {key: [] for key in variant_losses if key != "baseline"}

    for seed in seeds:
        torch.manual_seed(seed)
        initial_model = SmallCNNTanh()
        batches = make_batch_schedule(len(X), batch_size, epochs, seed)

        model_b = copy.deepcopy(initial_model)
        variant_losses["baseline"].append(
            train_minibatch_baseline(model_b, X, y, batches, loss_fn, lr=0.05, momentum=0.7)
        )

        model_naive = copy.deepcopy(initial_model)
        loss_naive, info_naive = train_minibatch_anderson_classical(
            model_naive, X, y, batches, loss_fn, lr=0.05, momentum=0.7,
            window=5, aa_interval=5, history_stride=1, safeguard_dataset="batch"
        )
        variant_losses["classical_batch_safeguard"].append(loss_naive)
        variant_infos["classical_batch_safeguard"].append(info_naive)

        model_full = copy.deepcopy(initial_model)
        loss_full, info_full = train_minibatch_anderson_classical(
            model_full, X, y, batches, loss_fn, lr=0.05, momentum=0.7,
            window=5, aa_interval=5, history_stride=1, safeguard_dataset="full"
        )
        variant_losses["classical_full_safeguard"].append(loss_full)
        variant_infos["classical_full_safeguard"].append(info_full)

        model_sparse = copy.deepcopy(initial_model)
        loss_sparse, info_sparse = train_minibatch_anderson_classical(
            model_sparse, X, y, batches, loss_fn, lr=0.05, momentum=0.7,
            window=5, aa_interval=10, history_stride=5, safeguard_dataset="full"
        )
        variant_losses["classical_sparse_full_safeguard"].append(loss_sparse)
        variant_infos["classical_sparse_full_safeguard"].append(info_sparse)

        model_ito = copy.deepcopy(initial_model)
        loss_ito, info_ito = train_minibatch_ito_xue(
            model_ito, X, y, batches, loss_fn, lr=0.05, momentum=0.7,
            window=5, aa_interval=10, snapshot_stride=5, safeguard=True
        )
        variant_losses["ito_xue_full_safeguard"].append(loss_ito)
        variant_infos["ito_xue_full_safeguard"].append(info_ito)

    results = {}
    for variant, losses in variant_losses.items():
        if variant == "baseline":
            continue
        analysis = paired_analysis(variant_losses["baseline"], losses)
        results[variant] = analysis
        print(format_results_table(analysis, title=f"Mini-batch SGD: Baseline vs {variant}"))

    write_experiment_result(
        output_path or os.path.join("results", "test_minibatch_summary.json"),
        experiment="Mini-batch SGD Anderson compatibility",
        config={
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": 0.05,
            "momentum": 0.7,
            "model": "SmallCNNTanh",
            "batch_schedule": "torch.randperm per epoch with per-seed generator, shared by every variant",
        },
        results=results,
        per_seed={**variant_losses, "variant_infos": variant_infos},
        notes=[
            "This script supersedes earlier two-variant mini-batch harnesses.",
            "The full-dataset safeguard is not a held-out validation safeguard.",
        ],
    )

    print("=" * 68)
    return results


if __name__ == "__main__":
    main()
