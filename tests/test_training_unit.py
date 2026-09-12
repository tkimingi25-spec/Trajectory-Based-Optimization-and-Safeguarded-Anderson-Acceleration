"""
Unit tests for src/training.py — training loops and paired comparison.

Tests verify:
- Baseline training reduces loss on a trivial problem
- Anderson training runs without error and returns correct info dict
- Safeguarded Anderson doesn't blow up
- Paired comparison runner works end-to-end
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import TinyMLP
from src.training import (
    run_paired_comparison,
    train_baseline,
    train_with_anderson,
    train_with_anderson_ito_xue,
)


class TestTrainBaseline:
    """Tests for the baseline SGD training loop."""

    def test_loss_decreases(self):
        """Loss should decrease on a simple regression task."""
        torch.manual_seed(42)
        model = TinyMLP()
        X = torch.randn(32, 4)
        y = torch.randn(32, 1)
        loss_fn = nn.MSELoss()

        initial_loss = loss_fn(model(X), y).item()
        final_loss, history = train_baseline(model, X, y, loss_fn, steps=20, lr=0.05, momentum=0.7)

        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_returns_correct_history_keys(self):
        """History dict should contain 'losses' and 'trajectory'."""
        model = TinyMLP()
        X = torch.randn(16, 4)
        y = torch.randn(16, 1)

        _, history = train_baseline(model, X, y, nn.MSELoss(), steps=5)

        assert "losses" in history
        assert "trajectory" in history
        assert len(history["losses"]) == 5

    def test_trajectory_logging(self):
        """When log_trajectory=True, trajectory should be populated."""
        model = TinyMLP()
        X = torch.randn(16, 4)
        y = torch.randn(16, 1)

        _, history = train_baseline(model, X, y, nn.MSELoss(), steps=5, log_trajectory=True)

        assert len(history["trajectory"]) == 5
        D = sum(p.numel() for p in model.parameters())
        assert history["trajectory"][0].shape == (D,)


class TestTrainWithAnderson:
    """Tests for safeguarded Anderson accelerated training."""

    def test_runs_without_error(self):
        """Anderson training should complete without exceptions."""
        torch.manual_seed(42)
        model = TinyMLP()
        X = torch.randn(32, 4)
        y = torch.randn(32, 1)

        final_loss, info = train_with_anderson(
            model,
            X,
            y,
            nn.MSELoss(),
            steps=15,
            lr=0.05,
            momentum=0.7,
            window=3,
            aa_interval=5,
            safeguard=True,
        )

        assert np.isfinite(final_loss)

    def test_returns_correct_info_keys(self):
        """Info dict should contain jump statistics."""
        model = TinyMLP()
        X = torch.randn(16, 4)
        y = torch.randn(16, 1)

        _, info = train_with_anderson(
            model,
            X,
            y,
            nn.MSELoss(),
            steps=15,
            window=3,
            aa_interval=5,
        )

        assert "losses" in info
        assert "jumps_attempted" in info
        assert "jumps_accepted" in info
        assert info["jumps_accepted"] <= info["jumps_attempted"]

    def test_safeguard_prevents_blowup(self):
        """With safeguard=True, final loss should remain finite."""
        torch.manual_seed(42)
        model = TinyMLP()
        X = torch.randn(32, 4)
        y = torch.randn(32, 1)

        final_loss, _ = train_with_anderson(
            model,
            X,
            y,
            nn.MSELoss(),
            steps=20,
            safeguard=True,
        )

        assert np.isfinite(final_loss), f"Loss blew up: {final_loss}"


class TestTrainWithAndersonItoXue:
    """Tests for Ito & Xue Anderson-type training."""

    def test_runs_without_error(self):
        """Ito-Xue training should complete without exceptions."""
        torch.manual_seed(42)
        model = TinyMLP()
        X = torch.randn(32, 4)
        y = torch.randn(32, 1)

        final_loss, info = train_with_anderson_ito_xue(
            model,
            X,
            y,
            nn.MSELoss(),
            steps=15,
            lr=0.05,
            momentum=0.7,
            window=3,
            aa_interval=5,
            safeguard=True,
            is_classification=False,
        )

        assert np.isfinite(final_loss)
        assert "jumps_attempted" in info
        assert "jumps_accepted" in info


class TestPairedComparison:
    """Tests for seed-matched paired benchmarking runner."""

    def test_runs_end_to_end(self):
        """Paired comparison should complete and return valid statistics."""

        def model_fn():
            return TinyMLP()

        def data_fn(seed):
            torch.manual_seed(seed)
            return torch.randn(16, 4), torch.randn(16, 1)

        results, base_losses, aa_losses = run_paired_comparison(
            model_fn=model_fn,
            get_data_fn=data_fn,
            loss_fn=nn.MSELoss(),
            seeds=range(100, 102),
            steps=10,
        )

        assert "n_seeds" in results
        assert results["n_seeds"] == 2
        assert len(base_losses) == 2
        assert len(aa_losses) == 2
        assert all(np.isfinite(loss_item) for loss_item in base_losses)
        assert all(np.isfinite(loss_item) for loss_item in aa_losses)
