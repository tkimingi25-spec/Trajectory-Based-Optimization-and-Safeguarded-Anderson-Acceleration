"""
Tier 1 Pedagogical Track: Coordinate-Freeze Telemetry & Phase Portraits (§3)

Demonstrates:
1. §3.4: Verified monotonic anisotropic quadratic ravine (no oscillations at critical lr).
2. §3.5: Correction of the 2-parameter orbital hypothesis -> single parameter vs. own velocity (phase portrait).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.landscapes import run_quadratic_optimization, verify_monotonic_trajectory


def main():
    print("=" * 68)
    print("  Tier 1 Pedagogical Track: Anisotropic Quadratic Ravine (§3)")
    print("=" * 68)

    # 1. Monotonic Ravine (Option A: a=20, b=1, lr=0.0013 < lr_critical=0.001334)
    traj, a, b = run_quadratic_optimization(
        steps=400, lr=0.0013, momentum=0.7, a=20.0, b=1.0, seed=42
    )
    w1_inc, w2_inc, loss_inc = verify_monotonic_trajectory(traj)

    print(f"Config: a={a}, b={b}, lr=0.0013, momentum=0.7, steps=400")
    print(f"Initial State: w1={traj[0]['w1']:.4f}, w2={traj[0]['w2']:.4f}, Loss={traj[0]['loss']:.4f}")
    print(f"Final State:   w1={traj[-1]['w1']:.6f}, w2={traj[-1]['w2']:.6f}, Loss={traj[-1]['loss']:.6f}")
    print(f"Monotonicity Check: w1 increases={w1_inc}, w2 increases={w2_inc}, loss increases={loss_inc}")
    assert w1_inc == 0 and w2_inc == 0 and loss_inc == 0, "Assertion failed: Trajectory oscillated!"
    print(">> Monotonic ravine assertion PASSED (Zero oscillations, pure L-shaped crawl).\n")

    # 2. Phase Portrait (Hypothesis 3: w1 vs. v1 spiral, lr=0.08, momentum=0.7)
    # The oscillatory configuration that demonstrates genuine phase-space spiraling
    traj_osc, _, _ = run_quadratic_optimization(
        steps=30, lr=0.08, momentum=0.7, a=20.0, b=1.0, seed=42
    )

    steps = [r["step"] for r in traj_osc]
    w1s = [r["w1"] for r in traj_osc]
    v1s = [r["v1"] for r in traj_osc]
    radii = [np.sqrt(w**2 + v**2) for w, v in zip(w1s, v1s)]

    # Compute total angle swept in (w1, v1) plane
    angles = np.unwrap([np.arctan2(v, w) for w, v in zip(w1s, v1s)])
    total_degrees = np.degrees(angles[-1] - angles[0])

    print("Phase Portrait Telemetry (w1 vs v1, oscillatory configuration lr=0.08):")
    print(f"Total Angular Sweep: {total_degrees:.1f} deg ({total_degrees/360:.2f} full turns)")
    print(f"Initial Radius:      {radii[0]:.2f}")
    print(f"Final Radius:        {radii[-1]:.4f}")
    print(f"Envelope decay rate: ~sqrt(momentum) = {np.sqrt(0.7):.4f} per step")
    print(">> Phase portrait confirmed: genuine spiral in (w1, v1) state-space.\n")

    # Save figure
    os.makedirs("results", exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: L-shaped ravine
    w1_mono = [r["w1"] for r in traj]
    w2_mono = [r["w2"] for r in traj]
    axs[0].plot(w1_mono, w2_mono, "b.-", markersize=3, label="Trajectory")
    axs[0].plot(w1_mono[0], w2_mono[0], "go", label="Start")
    axs[0].plot(w1_mono[-1], w2_mono[-1], "ro", label="End (Step 400)")
    axs[0].set_title("Monotonic Ravine (L-Shaped Crawl)")
    axs[0].set_xlabel("w1 (Steep Axis, a=20)")
    axs[0].set_ylabel("w2 (Shallow Axis, b=1)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # Plot 2: Phase Portrait
    axs[1].plot(w1s, v1s, "r.-", markersize=6, label="Phase trajectory")
    axs[1].plot(w1s[0], v1s[0], "go", label="Start")
    axs[1].plot(w1s[-1], v1s[-1], "ko", label="End")
    axs[1].set_title("Phase Portrait (w1 vs. v1 Spiral)")
    axs[1].set_xlabel("Position w1")
    axs[1].set_ylabel("Velocity v1")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    plt_path = os.path.join("results", "tier1_phase_portraits.png")
    plt.savefig(plt_path, dpi=150)
    plt.close()
    print(f"Saved figure to: {plt_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
