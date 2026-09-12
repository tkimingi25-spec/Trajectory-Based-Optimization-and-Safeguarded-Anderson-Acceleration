"""
Unit tests for src/statistics.py — paired statistical analysis.

Tests verify:
- Correct win/loss/tie counting
- p-value computation
- Significance detection
- Output format of results table
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.statistics import format_results_table, paired_analysis


class TestPairedAnalysis:
    """Tests for the paired_analysis function."""

    def test_clear_win(self):
        """When test is always better, should report all wins."""
        rng = np.random.default_rng(42)
        baseline = 1.0 + 0.05 * rng.standard_normal(30)
        test = 0.5 + 0.05 * rng.standard_normal(30)

        result = paired_analysis(baseline, test)

        assert result["n_seeds"] == 30
        assert result["wins"] == 30
        assert result["losses"] == 0
        assert result["ties"] == 0
        assert result["mean_diff"] < 0
        assert result["rel_reduction_pct"] > 0
        assert result["is_significant"] is True

    def test_clear_loss(self):
        """When test is always worse, should report all losses."""
        baseline = [0.5] * 30
        test = [1.0] * 30

        result = paired_analysis(baseline, test)

        assert result["wins"] == 0
        assert result["losses"] == 30
        assert result["mean_diff"] > 0

    def test_ties(self):
        """Identical results should be ties."""
        baseline = [1.0] * 30
        test = [1.0] * 30

        result = paired_analysis(baseline, test)

        assert result["ties"] == 30
        assert result["wins"] == 0
        assert result["losses"] == 0

    def test_mixed_results(self):
        """Mixed results should have partial wins/losses."""
        baseline = [1.0, 2.0, 3.0, 4.0, 5.0] * 6
        test = [0.5, 2.5, 2.5, 4.5, 4.5] * 6

        result = paired_analysis(baseline, test)

        assert result["wins"] + result["losses"] + result["ties"] == 30

    def test_length_mismatch_raises(self):
        """Should raise assertion on mismatched lengths."""
        with pytest.raises(AssertionError):
            paired_analysis([1.0, 2.0], [1.0])

    def test_relative_reduction_positive_for_improvement(self):
        """Relative reduction should be positive when test is better."""
        result = paired_analysis([1.0] * 30, [0.7] * 30)
        assert result["rel_reduction_pct"] == pytest.approx(30.0, abs=0.1)

    def test_pvalue_types(self):
        """p-values should be floats."""
        result = paired_analysis([1.0, 1.1, 0.9] * 10, [0.5, 0.6, 0.4] * 10)
        assert isinstance(result["p_t"], float)
        assert isinstance(result["p_w"], float)

    def test_result_keys(self):
        """Should contain all expected keys."""
        result = paired_analysis([1.0] * 30, [0.5] * 30)
        expected_keys = {
            "n_seeds",
            "mean_baseline",
            "std_baseline",
            "mean_test",
            "std_test",
            "mean_diff",
            "std_diff",
            "rel_reduction_pct",
            "wins",
            "losses",
            "ties",
            "t_stat",
            "p_t",
            "w_stat",
            "p_w",
            "is_significant",
        }
        assert set(result.keys()) == expected_keys


class TestFormatResultsTable:
    """Tests for the table formatter."""

    def test_returns_string(self):
        """Should return a formatted string."""
        result = paired_analysis([1.0] * 30, [0.5] * 30)
        table = format_results_table(result, title="Test Experiment")

        assert isinstance(table, str)
        assert "Test Experiment" in table
        assert "Seeds tested" in table

    def test_contains_verdict(self):
        """Output should include a verdict line."""
        result = paired_analysis([1.0] * 30, [0.5] * 30)
        table = format_results_table(result)
        assert "Verdict" in table
