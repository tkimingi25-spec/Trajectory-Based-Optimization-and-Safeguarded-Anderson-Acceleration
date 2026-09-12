"""
Unit tests for src/anderson.py — Anderson acceleration core algorithms.

Tests cover:
- State management (get/set round-trip consistency)
- Classical Type-II extrapolation correctness
- Ito & Xue formulation constraints
- Safeguard and sanity check utilities
"""

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
from src.models import TinyMLP


class TestStateManagement:
    """Tests for get/set full state and position-only round-trips."""

    def test_get_set_full_state_roundtrip(self):
        """Setting and getting full state should produce identical vectors."""
        model = TinyMLP()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.7)

        # Run one step to populate momentum buffers
        X = torch.randn(8, 4)
        y = torch.randn(8, 1)
        loss = nn.MSELoss()(model(X), y)
        loss.backward()
        optimizer.step()

        D = sum(p.numel() for p in model.parameters())
        state = get_full_state(model, optimizer, D)

        assert state.shape == (2 * D,), f"Expected shape ({2 * D},), got {state.shape}"

        # Create a fresh model and restore state
        model2 = TinyMLP()
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.05, momentum=0.7)
        set_full_state(model2, optimizer2, state, D)

        state2 = get_full_state(model2, optimizer2, D)
        np.testing.assert_allclose(state, state2, atol=1e-7)

    def test_get_set_position_only_roundtrip(self):
        """Setting and getting position should produce identical vectors."""
        model = TinyMLP()
        pos = get_position_only(model)
        D = sum(p.numel() for p in model.parameters())
        assert pos.shape == (D,)

        model2 = TinyMLP()
        set_position_only(model2, pos)
        pos2 = get_position_only(model2)
        np.testing.assert_allclose(pos, pos2, atol=1e-7)

    def test_full_state_dimension(self):
        """Full state should be 2*D (position + velocity)."""
        model = TinyMLP()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.7)
        D = sum(p.numel() for p in model.parameters())
        state = get_full_state(model, optimizer, D)
        assert len(state) == 2 * D


class TestClassicalAnderson:
    """Tests for classical Type-II Anderson acceleration."""

    def _make_simple_history(self, window=5, dim=10):
        """Create a simple convergent sequence for testing."""
        rng = np.random.default_rng(42)
        target = rng.standard_normal(dim)
        states = []
        x = rng.standard_normal(dim)
        for _i in range(window + 2):
            x = 0.5 * x + 0.5 * target + 0.01 * rng.standard_normal(dim)
            states.append(x.copy())
        return states

    def test_returns_correct_shape(self):
        """Extrapolated state should match input dimension."""
        states = self._make_simple_history(window=5, dim=10)
        result = anderson_extrapolate_classical(states, window=5)
        assert result is not None
        assert result.shape == (10,)

    def test_returns_none_insufficient_history(self):
        """Should return None when history is too short."""
        states = [np.random.randn(10)]
        result = anderson_extrapolate_classical(states, window=5)
        assert result is None

    def test_returns_none_for_two_states(self):
        """Need at least window+1 states for extrapolation."""
        states = [np.random.randn(10), np.random.randn(10)]
        result = anderson_extrapolate_classical(states, window=5)
        assert result is None

    def test_output_is_finite(self):
        """Extrapolated state should contain only finite values."""
        states = self._make_simple_history(window=3, dim=20)
        result = anderson_extrapolate_classical(states, window=3, reg=1e-6)
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_regularization_prevents_singular_gram(self):
        """With near-identical states, regularization should prevent crash."""
        base = np.ones(10)
        states = [base + 1e-10 * np.random.randn(10) for _ in range(7)]
        result = anderson_extrapolate_classical(states, window=5, reg=1e-4)
        # Should either return None or a finite result — not crash
        if result is not None:
            assert np.all(np.isfinite(result))


class TestItoXueAnderson:
    """Tests for Ito & Xue Anderson-type acceleration."""

    def test_returns_correct_shape(self):
        """Combined parameter vector should match input dimension."""
        rng = np.random.default_rng(42)
        dim = 10
        n_samples = 5
        param_snapshots = [rng.standard_normal(dim) for _ in range(3)]
        predictions = [rng.standard_normal((n_samples, 2)) for _ in range(3)]
        targets = rng.standard_normal((n_samples, 2))

        result = anderson_extrapolate_ito_xue(param_snapshots, predictions, targets)
        assert result is not None
        assert result.shape == (dim,)

    def test_returns_none_single_snapshot(self):
        """Need at least 2 snapshots."""
        result = anderson_extrapolate_ito_xue(
            [np.random.randn(10)],
            [np.random.randn(5, 2)],
            np.random.randn(5, 2),
        )
        assert result is None

    def test_output_is_finite(self):
        """Combined parameters should be finite."""
        rng = np.random.default_rng(42)
        param_snapshots = [rng.standard_normal(10) for _ in range(4)]
        predictions = [rng.standard_normal((8, 3)) for _ in range(4)]
        targets = rng.standard_normal((8, 3))

        result = anderson_extrapolate_ito_xue(param_snapshots, predictions, targets)
        if result is not None:
            assert np.all(np.isfinite(result))


class TestSafeguardUtilities:
    """Tests for safeguard and sanity check functions."""

    def test_strict_descent_accepts_improvement(self):
        assert passes_strict_descent_safeguard(0.5, 1.0) is True

    def test_strict_descent_rejects_worse(self):
        assert passes_strict_descent_safeguard(1.5, 1.0) is False

    def test_strict_descent_rejects_equal(self):
        assert passes_strict_descent_safeguard(1.0, 1.0) is False

    def test_strict_descent_rejects_nan(self):
        assert passes_strict_descent_safeguard(float("nan"), 1.0) is False

    def test_strict_descent_rejects_inf(self):
        assert passes_strict_descent_safeguard(float("inf"), 1.0) is False

    def test_state_is_sane_finite(self):
        assert state_is_sane(np.array([1.0, 2.0, 3.0])) is True

    def test_state_is_sane_rejects_nan(self):
        assert state_is_sane(np.array([1.0, float("nan"), 3.0])) is False

    def test_state_is_sane_rejects_extreme(self):
        assert state_is_sane(np.array([1.0, 1e5, 3.0]), max_abs=1e4) is False

    def test_state_is_sane_max_abs_configurable(self):
        assert state_is_sane(np.array([500.0]), max_abs=1000) is True
        assert state_is_sane(np.array([500.0]), max_abs=100) is False
