"""Master Experiment Runner & Verification Suite.

Executes the full suite of reproducible experiments:
1. Tier 1: Quadratic Ravine & Phase Portraits (§3)
2. Tier 2: Diagnostics, PCA & Shifted Power Iteration (§4)
3. Saddle Escape Negative Results (§5.1-§5.9)
4. Baldi-Hornik Saddles #1 & #2 Anderson Acceleration (§5.10)
5. Real Networks: TinyMLP, ReLU CNN, Tanh CNN (§5.11-§5.13)
6. Controlled activation smoothness ablation
7. Mini-batch SGD Failure Analysis (Test #2)
8. Late-Training Behavior & Horizon Dynamics (Test #3)

Supports `--quick` mode for rapid smoke testing in CI or development.
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.logging_config import setup_logger

logger = setup_logger("master_runner", log_file="master_run.log")


def run_command(cmd, desc):
    """Execute a subprocess command with timer and status logging."""
    print("\n" + "#" * 72)
    print(f"#  RUNNING: {desc}")
    print("#" * 72 + "\n")
    logger.info("Starting: %s", desc)

    start = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - start

    print(f"\n[Completed in {elapsed:.2f}s]")
    logger.info("Completed: %s in %.2fs", desc, elapsed)


def run_module(module_name, desc):
    """Execute a Python module in the experiments directory."""
    script_path = os.path.join(os.path.dirname(__file__), module_name)
    run_command([sys.executable, script_path], desc)


def main():
    """CLI entry point for running experiment suites."""
    parser = argparse.ArgumentParser(description="Master Anderson acceleration experiment runner")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast smoke-test mode (<30s) suitable for CI verification",
    )
    args = parser.parse_args()

    if args.quick:
        print("=" * 72)
        print("  ANDERSON ACCELERATION RESEARCH: QUICK SMOKE-TEST SUITE")
        print("=" * 72)
        logger.info("Starting quick smoke-test suite")

        run_module("tier1_ravine.py", "Tier 1: Monotonic Ravine & Phase Portraits (§3)")
        run_module("tier2_diagnostics.py", "Tier 2: Trajectory PCA & Hessian Diagnostics (§4)")
        run_module("test_saddle_escape.py", "Curvature-Informed Negative Results (§5.1-§5.9)")
        run_command(
            [
                sys.executable,
                "-c",
                "from experiments.test_smoothness_ablation import main; main(n_seeds=2, steps=5, lrs=(0.05,))",
            ],
            "Quick Smoothness Ablation Smoke Test (2 seeds, 5 steps)",
        )

        print("\n" + "=" * 72)
        print("  QUICK SMOKE-TEST SUITE COMPLETED SUCCESSFULLY.")
        print("=" * 72)
        logger.info("Quick smoke-test suite completed successfully")
        return

    print("=" * 72)
    print("  ANDERSON ACCELERATION RESEARCH: COMPLETE EXPERIMENTAL SUITE")
    print("=" * 72)
    logger.info("Starting complete experimental suite")

    run_module("tier1_ravine.py", "Tier 1: Monotonic Ravine & Phase Portraits (§3)")
    run_module("tier2_diagnostics.py", "Tier 2: Trajectory PCA & Hessian Diagnostics (§4)")
    run_module("test_saddle_escape.py", "Curvature-Informed Negative Results (§5.1-§5.9)")
    run_module("test_anderson_saddles.py", "Baldi-Hornik Saddles #1 & #2 (§5.10)")
    run_module("test_anderson_real.py", "Real Networks & Smoothness Isolation (§5.11-§5.13)")
    run_module("test_smoothness_ablation.py", "Controlled Activation Smoothness Ablation")
    run_module("test_minibatch.py", "Mini-Batch SGD Evaluation (Test #2)")
    run_module("test_late_training.py", "Late-Training Dynamics & Horizons (Test #3)")

    print("\n" + "=" * 72)
    print("  ALL EXPERIMENTAL MODULES COMPLETED SUCCESSFULLY.")
    print("=" * 72)
    logger.info("All experimental modules completed successfully")


if __name__ == "__main__":
    main()
