"""
Shared pytest fixtures for Anderson acceleration test suite.

Provides lightweight fixtures with reduced seeds/steps for CI speed.
Full experiment validation uses the experiments/ scripts directly.
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import get_digits_data
from src.landscapes import make_baldi_hornik_saddle
from src.models import SmallCNN, SmallCNNTanh, TinyMLP

# ---------------------------------------------------------------------------
# Seed control
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def set_deterministic_seeds():
    """Fix random seeds for reproducibility in every test."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield


# ---------------------------------------------------------------------------
# CI-fast parameters (reduced seeds/steps for quick runs)
# ---------------------------------------------------------------------------

CI_SEEDS = range(1000, 1002)  # 2 seeds
CI_STEPS = 5  # 5 training steps


@pytest.fixture
def ci_seeds():
    return CI_SEEDS


@pytest.fixture
def ci_steps():
    return CI_STEPS


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_mlp():
    """Fresh TinyMLP instance (D=49)."""
    torch.manual_seed(42)
    return TinyMLP()


@pytest.fixture
def small_cnn():
    """Fresh SmallCNN (ReLU) instance."""
    torch.manual_seed(42)
    return SmallCNN()


@pytest.fixture
def small_cnn_tanh():
    """Fresh SmallCNNTanh instance."""
    torch.manual_seed(42)
    return SmallCNNTanh()


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def digits_data():
    """Sklearn digits dataset as tensors."""
    return get_digits_data()


@pytest.fixture
def tiny_mlp_data():
    """Small synthetic data for TinyMLP (input_dim=4)."""
    torch.manual_seed(42)
    X = torch.randn(32, 4)
    y = torch.randn(32, 1)
    return X, y


@pytest.fixture
def saddle_landscape_1():
    """Baldi-Hornik Saddle #1 (D=24)."""
    return make_baldi_hornik_saddle(saddle_type=1)


# ---------------------------------------------------------------------------
# Loss function fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mse_loss():
    return nn.MSELoss()


@pytest.fixture
def ce_loss():
    return nn.CrossEntropyLoss()
