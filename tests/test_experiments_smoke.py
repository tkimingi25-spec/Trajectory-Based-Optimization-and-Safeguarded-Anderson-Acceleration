"""
Smoke tests wrapping the experiment scripts in experiments/.

These tests call each experiment's main() function with reduced seeds/steps
to verify they don't crash. They do NOT validate full statistical results —
that's the job of the full experiment suite.

Marked as slow to exclude from default CI runs.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.mark.slow
class TestExperimentSmoke:
    """Smoke tests for experiment scripts."""

    def test_smoothness_ablation_smoke(self, tmp_path):
        """Smoothness ablation script should run with minimal params."""
        from experiments.test_smoothness_ablation import main

        out_file = str(tmp_path / "smoothness_smoke.json")
        results = main(
            n_seeds=2,
            seed_start=9000,
            lrs=(0.05,),
            steps=5,
            output_path=out_file,
        )
        assert results is not None
        assert len(results) > 0

    def test_minibatch_import(self):
        """Minibatch experiment script should be importable."""
        from experiments import test_minibatch

        assert hasattr(test_minibatch, "main")

    def test_late_training_import(self):
        """Late training experiment script should be importable."""
        from experiments import test_late_training

        assert hasattr(test_late_training, "main")

    def test_anderson_saddles_import(self):
        """Anderson saddles experiment script should be importable."""
        from experiments import test_anderson_saddles

        assert hasattr(test_anderson_saddles, "main")

    def test_anderson_real_import(self):
        """Anderson real networks experiment script should be importable."""
        from experiments import test_anderson_real

        assert hasattr(test_anderson_real, "main")
