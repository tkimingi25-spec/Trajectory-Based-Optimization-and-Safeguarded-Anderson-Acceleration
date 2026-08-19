"""
Master Experiment Runner & Verification Suite

Executes the full suite of reproducible experiments:
1. Tier 1: Quadratic Ravine & Phase Portraits (§3)
2. Tier 2: Diagnostics, PCA & Shifted Power Iteration (§4)
3. Saddle Escape Negative Results (§5.1-§5.9)
4. Baldi-Hornik Saddles #1 & #2 Anderson Acceleration (§5.10)
5. Real Networks: TinyMLP, ReLU CNN, Tanh CNN (§5.11-§5.13)
6. Controlled activation smoothness ablation
7. Mini-batch SGD Failure Analysis (Test #2)
8. Late-Training Behavior & Horizon Dynamics (Test #3)
"""

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_module(module_name, desc):
    print("\n" + "#" * 72)
    print(f"#  RUNNING: {desc}")
    print("#" * 72 + "\n")
    start = time.time()
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), module_name)], check=True)
    elapsed = time.time() - start
    print(f"\n[Completed in {elapsed:.2f}s]")


def main():
    print("=" * 72)
    print("  ANDERSON ACCELERATION RESEARCH: COMPLETE EXPERIMENTAL SUITE")
    print("=" * 72)

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


if __name__ == "__main__":
    main()
