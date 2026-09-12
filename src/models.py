"""
Neural network architectures used throughout the investigation.

All architectures are intentionally small (D=24 to D~9800 parameters)
to enable exhaustive, many-seed statistical testing on CPU.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# TinyMLP — §4.3, §5.11 (D=49 parameters)
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
    """
    Tiny regression MLP: Linear(4,8) -> tanh -> Linear(8,1).
    D = 4*8 + 8 + 8*1 + 1 = 49 parameters.
    Used for the first real-network validation of trajectory PCA,
    saddle detection, and Anderson acceleration.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# SmallCNN — §5.12 (ReLU, D~9802 parameters, null result)
# ---------------------------------------------------------------------------


class SmallCNN(nn.Module):
    """
    Small classifier CNN with ReLU activations.
    Conv2d(1,8,3,pad=1) -> Conv2d(8,16,3,pad=1) -> MaxPool(2)
    -> Linear(16*4*4, 32) -> Linear(32, 10).

    On sklearn digits (8x8 images):
        After conv1: 8x8x8
        After conv2: 8x8x16
        After pool:  4x4x16 = 256
        fc1: 256->32, fc2: 32->10
        D = (1*8*3*3+8) + (8*16*3*3+16) + (256*32+32) + (32*10+10)
          = 80 + 1168 + 8224 + 330 = 9802
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# SmallCNNTanh — §5.13 (tanh, same architecture; historical positive result)
# ---------------------------------------------------------------------------


class SmallCNNTanh(nn.Module):
    """
    Identical architecture to SmallCNN but with tanh activations.
    The ONLY change is relu -> tanh throughout.

    This supports the activation-smoothness hypothesis when paired with a
    same-seed, same-learning-rate ablation script.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# LinearAutoencoder — §5.3 (Baldi-Hornik strict saddle construction)
# ---------------------------------------------------------------------------


class LinearAutoencoder(nn.Module):
    """
    Linear autoencoder: enc (d -> k, no bias) -> dec (k -> d, no bias).
    Per Baldi & Hornik (1989), critical points of the MSE loss on this
    architecture are projections onto k eigenvectors of the input covariance.
    The global optimum uses the top-k; any other k-subset is a strict saddle.
    """

    def __init__(self, d, k):
        super().__init__()
        self.enc = nn.Linear(d, k, bias=False)
        self.dec = nn.Linear(k, d, bias=False)

    def forward(self, x):
        return self.dec(self.enc(x))


def count_parameters(model):
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clone_model(model):
    """Create a deep copy of a model with identical weights."""
    import copy

    return copy.deepcopy(model)
