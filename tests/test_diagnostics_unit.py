"""
Unit tests for src/diagnostics.py — trajectory PCA, Hessian-vector products, and guards.

Tests verify:
- PCA explained variance sums to 1.0
- Minimum eigenvalue estimation recovers known values
- ExplosionGuard triggers correctly
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.diagnostics import ExplosionGuard, estimate_lambda_min, trajectory_pca


class TestTrajectoryPCA:
    """Tests for trajectory PCA (Engine A)."""

    def test_variance_sums_to_one(self):
        """Explained variance ratios should sum to approximately 1.0."""
        rng = np.random.default_rng(42)
        # Create a trajectory matrix (T steps, D dimensions)
        trajectory = rng.standard_normal((50, 20))
        var_ratio, components, projections = trajectory_pca(trajectory)

        assert var_ratio.sum() == pytest.approx(1.0, abs=1e-6)

    def test_output_shapes(self):
        """PCA outputs should have correct dimensions."""
        trajectory = np.random.randn(30, 10)
        var_ratio, components, projections = trajectory_pca(trajectory)

        assert var_ratio.shape == (10,)
        assert components.shape[0] == 10  # min(T, D) components
        assert projections.shape[0] == 30  # T projections

    def test_dominant_component_captures_variance(self):
        """When trajectory lies mostly along one axis, PC1 should capture most variance."""
        rng = np.random.default_rng(42)
        # Create trajectory along axis 0 with small noise
        trajectory = np.zeros((50, 10))
        trajectory[:, 0] = np.linspace(0, 10, 50)
        trajectory += 0.01 * rng.standard_normal((50, 10))

        var_ratio, _, _ = trajectory_pca(trajectory)

        assert var_ratio[0] > 0.95, f"PC1 should capture >95% variance, got {var_ratio[0]:.4f}"

    def test_handles_constant_trajectory(self):
        """Should handle degenerate case where trajectory is constant."""
        trajectory = np.ones((20, 5))
        var_ratio, _, _ = trajectory_pca(trajectory)
        # All variance should be zero
        assert np.allclose(var_ratio, 0.0)


class TestEstimateLambdaMin:
    """Tests for the shifted power iteration minimum eigenvalue estimator."""

    def test_recovers_known_eigenvalue(self):
        """Should recover the minimum eigenvalue of a known matrix."""
        # Construct a simple symmetric matrix with known eigenvalues
        # Eigenvalues: -2, 1, 3
        eigenvalues = np.array([-2.0, 1.0, 3.0])
        Q = np.eye(3)  # Eigenvectors are identity (diagonal matrix)
        H = np.diag(eigenvalues)

        def apply_hv(v):
            return H @ v

        lambda_min, v_min = estimate_lambda_min(apply_hv, D=3, iters=500, seed=1)

        assert lambda_min == pytest.approx(-2.0, abs=0.1), (
            f"Expected lambda_min ≈ -2.0, got {lambda_min:.4f}"
        )

    def test_positive_definite_matrix(self):
        """For a positive definite matrix, lambda_min should be positive."""
        H = np.diag([1.0, 2.0, 5.0])

        def apply_hv(v):
            return H @ v

        lambda_min, _ = estimate_lambda_min(apply_hv, D=3, iters=300, seed=1)
        assert lambda_min > 0


class TestExplosionGuard:
    """Tests for the step explosion guard."""

    def test_passes_normal_step(self):
        """Should pass when step is below threshold."""
        guard = ExplosionGuard(max_step_norm=3.0)
        prev = np.array([1.0, 2.0, 3.0])
        curr = np.array([1.1, 2.1, 3.1])

        assert guard.check(0, prev, curr) is True
        assert guard.triggered is False

    def test_triggers_on_explosion(self):
        """Should trigger when step exceeds threshold."""
        guard = ExplosionGuard(max_step_norm=1.0)
        prev = np.zeros(3)
        curr = np.array([10.0, 10.0, 10.0])

        assert guard.check(5, prev, curr) is False
        assert guard.triggered is True
        assert guard.trigger_step == 5

    def test_records_trigger_delta(self):
        """Should record the delta that triggered the guard."""
        guard = ExplosionGuard(max_step_norm=1.0)
        prev = np.zeros(3)
        curr = np.array([5.0, 0.0, 0.0])

        guard.check(10, prev, curr)
        assert guard.trigger_delta == pytest.approx(5.0, abs=1e-6)
