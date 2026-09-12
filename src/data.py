"""
Data utilities for all experiments.

Provides dataset generators for:
1. Synthetic quadratic / Baldi-Hornik data
2. Synthetic nonlinear regression data for TinyMLP (§4.3, §5.11)
3. Scikit-learn Digits dataset for CNN classification (§5.12, §5.13, Test #2, Test #3)
"""

import numpy as np
import torch
from sklearn.datasets import load_digits


def get_digits_data(train_ratio=1.0, seed=42):
    """
    Load the scikit-learn digits dataset (1797 images, 8x8, 10 classes).

    Standard full-batch configuration from §5.12:
    Images are shaped (N, 1, 8, 8) and normalized.

    Args:
        train_ratio: fraction of dataset to use for training (1.0 = full dataset)
        seed: random seed for shuffling if split is requested

    Returns:
        X (torch.Tensor): shape (N, 1, 8, 8), float32
        y (torch.Tensor): shape (N,), int64
    """
    digits = load_digits()
    X = digits.images.astype(np.float32)  # shape: (1797, 8, 8)
    y = digits.target.astype(np.int64)

    # Reshape to (N, 1, 8, 8) for PyTorch Conv2d
    X = X[:, np.newaxis, :, :]

    # Normalize to zero mean, unit variance
    X_mean = X.mean()
    X_std = X.std() + 1e-8
    X = (X - X_mean) / X_std

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    if train_ratio < 1.0:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(X_t))
        split_idx = int(len(X_t) * train_ratio)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        return (X_t[train_idx], y_t[train_idx]), (X_t[test_idx], y_t[test_idx])

    return X_t, y_t


def get_tinymlp_data(N=200, seed=123):
    """
    Synthetic nonlinear regression data for TinyMLP (§4.3, §5.11):
    y = sin(x1) + x2*x3 - x4^2

    Args:
        N: number of sample points (default: 200)
        seed: random seed for data generation (default: 123)

    Returns:
        X (torch.Tensor): shape (N, 4), float32
        y (torch.Tensor): shape (N, 1), float32
    """
    torch.manual_seed(seed)
    X = torch.randn(N, 4)
    y = (torch.sin(X[:, 0]) + X[:, 1] * X[:, 2] - X[:, 3] ** 2).unsqueeze(1)
    return X, y


def get_baldi_hornik_data(saddle_type=1, N=4000):
    """
    Generate synthetic data for Baldi-Hornik linear autoencoder saddles (§5.3).

    Saddle #1: d=6, variances=[10.0, 8.0, 5.0, 3.0, 1.0, 0.5], seed=7
    Saddle #2: d=10, variances=[12.0, 9.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5], seed=11

    Returns:
        X (torch.Tensor): shape (N, d), float32
        d (int): input dimension
        k (int): bottleneck dimension
        saddle_idx (list): planted non-optimal eigenvector indices
        opt_idx (list): true optimal eigenvector indices
        variances (np.ndarray): true covariance eigenvalues
    """
    if saddle_type == 1:
        d, k = 6, 2
        variances = np.array([10.0, 8.0, 5.0, 3.0, 1.0, 0.5])
        rng = np.random.default_rng(7)
        saddle_idx = [2, 3]  # Non-optimal: indices 2,3 instead of 0,1
        opt_idx = [0, 1]
    elif saddle_type == 2:
        d, k = 10, 3
        variances = np.array([12.0, 9.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5])
        rng = np.random.default_rng(11)
        saddle_idx = [0, 2, 4]  # Non-optimal: indices 0,2,4 instead of 0,1,2
        opt_idx = [0, 1, 2]
    else:
        raise ValueError(f"Unknown saddle type: {saddle_type}")

    X_np = rng.standard_normal((N, d)) * np.sqrt(variances)
    X = torch.tensor(X_np, dtype=torch.float32)
    return X, d, k, saddle_idx, opt_idx, variances
