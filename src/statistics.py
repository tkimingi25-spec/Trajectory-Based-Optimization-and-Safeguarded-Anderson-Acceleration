"""
Statistical analysis tools for optimizer comparisons.

Follows the strict statistical standard established in §2.3:
1. Exact per-seed win/loss/tie counts (matched initialization)
2. Parametric paired t-test (scipy.stats.ttest_1samp on differences)
3. Non-parametric Wilcoxon signed-rank test (robust to non-normality/outliers)
4. Well-formatted summary tables with mean, std, p-values, effect sizes
"""

import numpy as np
from scipy import stats


def paired_analysis(baseline_losses, test_losses, alpha=0.05):
    """
    Perform rigorous paired comparison between baseline and an accelerated method.

    Args:
        baseline_losses: list or 1D array of final losses from baseline
        test_losses: list or 1D array of final losses from accelerated method
        alpha: significance threshold (default: 0.05)

    Returns:
        dict containing comprehensive statistical metrics
    """
    b = np.asarray(baseline_losses, dtype=np.float64)
    t = np.asarray(test_losses, dtype=np.float64)
    n = len(b)
    assert len(t) == n, f"Length mismatch: {len(b)} vs {len(t)}"

    diffs = t - b  # negative difference means test method achieved lower loss (better)
    wins = int(np.sum(diffs < -1e-9))
    losses = int(np.sum(diffs > 1e-9))
    ties = int(np.sum(np.abs(diffs) <= 1e-9))

    mean_baseline = float(np.mean(b))
    std_baseline = float(np.std(b, ddof=1)) if n > 1 else 0.0
    mean_test = float(np.mean(t))
    std_test = float(np.std(t, ddof=1)) if n > 1 else 0.0

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    rel_reduction_pct = float((mean_baseline - mean_test) / (mean_baseline + 1e-12) * 100)

    # Paired t-test
    if n > 1 and std_diff > 1e-15:
        t_stat, p_t = stats.ttest_1samp(diffs, 0.0)
        t_stat, p_t = float(t_stat), float(p_t)
    else:
        t_stat, p_t = 0.0, 1.0

    # Wilcoxon signed-rank test
    non_zero_diffs = diffs[np.abs(diffs) > 1e-9]
    if len(non_zero_diffs) >= 10:
        try:
            w_stat, p_w = stats.wilcoxon(diffs, zero_method="pratt")
            w_stat, p_w = float(w_stat), float(p_w)
        except Exception:
            w_stat, p_w = float("nan"), 1.0
    else:
        w_stat, p_w = float("nan"), 1.0

    is_significant = (p_t < alpha) and (p_w < alpha if not np.isnan(p_w) else p_t < alpha)

    return {
        "n_seeds": n,
        "mean_baseline": mean_baseline,
        "std_baseline": std_baseline,
        "mean_test": mean_test,
        "std_test": std_test,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "rel_reduction_pct": rel_reduction_pct,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "t_stat": t_stat,
        "p_t": p_t,
        "w_stat": w_stat,
        "p_w": p_w,
        "is_significant": is_significant,
    }


def format_results_table(results_dict, title="Experiment Results"):
    """
    Format paired analysis dictionary into a clean textual summary table.
    """
    r = results_dict
    lines = [
        "=" * 68,
        f"  {title}",
        "=" * 68,
        f"Seeds tested:             {r['n_seeds']}",
        f"Baseline Loss (mean+/-std): {r['mean_baseline']:.6f} +/- {r['std_baseline']:.6f}",
        f"Test Loss (mean+/-std):     {r['mean_test']:.6f} +/- {r['std_test']:.6f}",
        f"Mean Difference:          {r['mean_diff']:+.6f} ({r['rel_reduction_pct']:+.2f}%)",
        f"Win / Loss / Tie:         {r['wins']} / {r['losses']} / {r['ties']} (Win rate: {r['wins']/r['n_seeds']*100:.1f}%)",
        f"Paired t-test:            t = {r['t_stat']:.4f}, p = {r['p_t']:.4e}",
        (
            f"Wilcoxon signed-rank:     p = {r['p_w']:.4e}"
            if not np.isnan(r["p_w"])
            else "Wilcoxon: N/A (<10 non-zero diffs)"
        ),
        "-" * 68,
        f"Verdict:                  {'STATISTICALLY SIGNIFICANT BENEFIT' if (r['is_significant'] and r['mean_diff'] < 0) else ('SIGNIFICANT DEGRADATION' if (r['is_significant'] and r['mean_diff'] > 0) else 'NULL RESULT (No significant difference)')}",
        "=" * 68,
    ]
    return "\n".join(lines)
