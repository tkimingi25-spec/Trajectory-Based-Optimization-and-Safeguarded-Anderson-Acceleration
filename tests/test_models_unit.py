"""
Unit tests for src/models.py — neural network architectures.

Tests verify:
- Parameter counts match documented values
- Forward pass output shapes are correct
- Model cloning produces independent copies
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import (
    LinearAutoencoder,
    SmallCNN,
    SmallCNNTanh,
    TinyMLP,
    clone_model,
    count_parameters,
)


class TestTinyMLP:
    """Tests for the TinyMLP architecture."""

    def test_parameter_count(self):
        """TinyMLP should have exactly 49 parameters."""
        model = TinyMLP()
        assert count_parameters(model) == 49

    def test_forward_shape(self):
        """Output should be (batch_size, 1)."""
        model = TinyMLP()
        X = torch.randn(16, 4)
        out = model(X)
        assert out.shape == (16, 1)

    def test_deterministic_with_seed(self):
        """Same seed should produce same initial weights."""
        torch.manual_seed(42)
        m1 = TinyMLP()
        torch.manual_seed(42)
        m2 = TinyMLP()

        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)


class TestSmallCNN:
    """Tests for the SmallCNN (ReLU) architecture."""

    def test_parameter_count(self):
        """SmallCNN should have 9802 parameters."""
        model = SmallCNN()
        assert count_parameters(model) == 9802

    def test_forward_shape(self):
        """Output should be (batch_size, 10) for 10-class classification."""
        model = SmallCNN()
        X = torch.randn(8, 1, 8, 8)
        out = model(X)
        assert out.shape == (8, 10)


class TestSmallCNNTanh:
    """Tests for the SmallCNNTanh (smooth) architecture."""

    def test_parameter_count(self):
        """SmallCNNTanh should have same parameter count as SmallCNN."""
        relu_model = SmallCNN()
        tanh_model = SmallCNNTanh()
        assert count_parameters(tanh_model) == count_parameters(relu_model)

    def test_forward_shape(self):
        """Output should be (batch_size, 10)."""
        model = SmallCNNTanh()
        X = torch.randn(8, 1, 8, 8)
        out = model(X)
        assert out.shape == (8, 10)


class TestLinearAutoencoder:
    """Tests for the Baldi-Hornik linear autoencoder."""

    def test_parameter_count(self):
        """LinearAutoencoder(d=6, k=2) should have 2*6*2 = 24 parameters."""
        model = LinearAutoencoder(d=6, k=2)
        assert count_parameters(model) == 24

    def test_forward_shape(self):
        """Output should match input dimension (autoencoder)."""
        model = LinearAutoencoder(d=6, k=2)
        X = torch.randn(10, 6)
        out = model(X)
        assert out.shape == (10, 6)


class TestCloneModel:
    """Tests for the model cloning utility."""

    def test_clone_is_independent(self):
        """Modifying clone should not affect original."""
        model = TinyMLP()
        cloned = clone_model(model)

        # Modify clone
        with torch.no_grad():
            for p in cloned.parameters():
                p.add_(1.0)

        # Original should be unchanged
        for p_orig, p_clone in zip(model.parameters(), cloned.parameters()):
            assert not torch.allclose(p_orig, p_clone)

    def test_clone_has_same_initial_weights(self):
        """Clone should start with identical weights."""
        model = TinyMLP()
        cloned = clone_model(model)

        for p_orig, p_clone in zip(model.parameters(), cloned.parameters()):
            assert torch.allclose(p_orig, p_clone)
