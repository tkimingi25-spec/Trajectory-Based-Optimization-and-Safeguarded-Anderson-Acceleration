"""Experiment configuration system with dataclasses and YAML loader."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AndersonConfig:
    """Hyperparameters for Anderson acceleration."""

    window: int = 5
    aa_interval: int = 10
    reg: float = 1e-6
    safeguard: bool = True
    safeguard_max_abs: float = 1e4


@dataclass
class OptimizerConfig:
    """Hyperparameters for base optimizer."""

    lr: float = 0.05
    momentum: float = 0.7
    weight_decay: float = 0.0


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str = "default_experiment"
    model: str = "SmallCNNTanh"
    n_seeds: int = 30
    seed_start: int = 7000
    steps: int = 100
    dataset: str = "digits"
    device: str = "cpu"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    anderson: AndersonConfig = field(default_factory=AndersonConfig)
    output_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Instantiate configuration from dictionary."""
        opt_data = data.pop("optimizer", {})
        aa_data = data.pop("anderson", {})
        return cls(
            optimizer=OptimizerConfig(**opt_data) if opt_data else OptimizerConfig(),
            anderson=AndersonConfig(**aa_data) if aa_data else AndersonConfig(),
            **data,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: str | Path | None = None, quick: bool = False) -> ExperimentConfig:
    """Load configuration from file or default, with optional --quick overrides."""
    if config_path and Path(config_path).is_file():
        cfg = ExperimentConfig.from_yaml(config_path)
    else:
        cfg = ExperimentConfig()

    if quick:
        cfg.n_seeds = 2
        cfg.steps = 5

    return cfg
