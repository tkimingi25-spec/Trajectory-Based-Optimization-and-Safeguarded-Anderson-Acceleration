"""
Generate publication-quality diagrams for the PDF report:
1. Smoothness vs Piecewise-Linear Activation Landscape
2. Safeguarded Concatenated Anderson Architecture Flowchart
3. Industry Application & Impact Taxonomy
"""

import os

import matplotlib.pyplot as plt
import numpy as np

os.makedirs("results", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# ---------------------------------------------------------------------------
# Diagram 1: Smoothness vs. Piecewise-Linear Loss Landscape
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

# (A) Smooth Landscape (Tanh / GELU)
x_smooth = np.linspace(-3, 3, 300)
y_smooth = x_smooth**4 - 3 * x_smooth**2 + 0.5 * x_smooth + 3
axs[0].plot(x_smooth, y_smooth, "b-", linewidth=2.5, label="Smooth Loss Surface L(w) (Tanh / GELU)")

# Iterates and secant
x_pts = np.array([-2.2, -1.8, -1.3, -0.6])
y_pts = x_pts**4 - 3 * x_pts**2 + 0.5 * x_pts + 3
axs[0].plot(x_pts, y_pts, "ro", markersize=7, label="Optimizer Iterates w_k")
for i in range(len(x_pts) - 1):
    axs[0].annotate(
        "",
        xy=(x_pts[i + 1], y_pts[i + 1]),
        xytext=(x_pts[i], y_pts[i]),
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
    )

# Anderson jump
x_jump = 1.3
y_jump = x_jump**4 - 3 * x_jump**2 + 0.5 * x_jump + 3
axs[0].annotate(
    "Anderson Secant Jump\n(Cuts across ravine)",
    xy=(x_jump, y_jump),
    xytext=(-0.5, 7.5),
    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2.5, ls="--"),
    bbox=dict(boxstyle="round,pad=0.4", fc="lightgreen", ec="darkgreen", alpha=0.8),
    fontsize=9,
    weight="bold",
)
axs[0].plot([x_jump], [y_jump], "g*", markersize=14, label="Extrapolated Iterate w_AA")

axs[0].set_title(
    "(A) Smooth C^infinity Landscape (Tanh / GELU)\nSecant extrapolations predict continuous trajectory",
    fontsize=11,
    weight="bold",
    pad=10,
)
axs[0].set_xlabel("Parameter Coordinate w", fontsize=10)
axs[0].set_ylabel("Loss Value L(w)", fontsize=10)
axs[0].legend(loc="upper right", fontsize=8.5)
axs[0].grid(True, alpha=0.3)

# (B) Piecewise-Linear Landscape (ReLU)
x_relu = np.linspace(-3, 3, 300)
# Construct piecewise linear
y_relu = np.piecewise(
    x_relu,
    [x_relu < -1.5, (x_relu >= -1.5) & (x_relu < 0), (x_relu >= 0) & (x_relu < 1.5), x_relu >= 1.5],
    [lambda x: -2.5 * x + 2, lambda x: -0.8 * x + 4.55, lambda x: 1.8 * x + 4.55, lambda x: 3.5 * x + 2.0],
)
axs[1].plot(x_relu, y_relu, "r-", linewidth=2.5, label="Piecewise-Linear Surface L(w) (ReLU)")
axs[1].axvline(x=-1.5, color="gray", linestyle=":", label="Activation Switching Boundaries")
axs[1].axvline(x=0.0, color="gray", linestyle=":")
axs[1].axvline(x=1.5, color="gray", linestyle=":")

# Broken Secant
axs[1].plot([-2.0, -1.2], [-2.5 * (-2.0) + 2, -0.8 * (-1.2) + 4.55], "ko", markersize=7)
axs[1].annotate(
    "Secant crosses kink\n(Invalid Linear Model)",
    xy=(1.0, 7.0),
    xytext=(-1.5, 9.0),
    arrowprops=dict(arrowstyle="->", color="red", lw=2, ls=":"),
    bbox=dict(boxstyle="round,pad=0.4", fc="#ffcccc", ec="red", alpha=0.8),
    fontsize=9,
    weight="bold",
)

axs[1].set_title(
    "(B) Piecewise-Linear Landscape (ReLU)\nNon-smooth kinks invalidate secant differences",
    fontsize=11,
    weight="bold",
    pad=10,
)
axs[1].set_xlabel("Parameter Coordinate w", fontsize=10)
axs[1].set_ylabel("Loss Value L(w)", fontsize=10)
axs[1].legend(loc="upper right", fontsize=8.5)
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
diag1_path = os.path.join("results", "diagram_smoothness_vs_relu.png")
plt.savefig(diag1_path)
plt.close()
print(f"Generated: {diag1_path}")

# ---------------------------------------------------------------------------
# Diagram 2: Architecture of Safeguarded Anderson Acceleration
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
ax.axis("off")

# Box styles
box_blue = dict(boxstyle="round,pad=0.5", fc="#e6f2ff", ec="#0066cc", lw=1.5)
box_green = dict(boxstyle="round,pad=0.5", fc="#e6ffe6", ec="#009933", lw=1.5)
box_red = dict(boxstyle="round,pad=0.5", fc="#ffe6e6", ec="#cc0000", lw=1.5)
box_gold = dict(boxstyle="round,pad=0.5", fc="#fff9e6", ec="#cc9900", lw=1.5)

ax.text(
    0.12,
    0.75,
    "Step 1: State History\ns_t = [w_t; v_t] in R^(2D)\nStores (m+1) States",
    ha="center",
    va="center",
    bbox=box_blue,
    fontsize=9,
    weight="bold",
)
ax.text(
    0.40,
    0.75,
    "Step 2: Type-II Least Squares\nmin ||F_diff * gamma - f_m||^2\nSolve m x m linear system",
    ha="center",
    va="center",
    bbox=box_gold,
    fontsize=9,
    weight="bold",
)
ax.text(
    0.72,
    0.75,
    "Step 3: State Extrapolation\ns_jump = sum alpha_i * s_i\nJoint w & v update",
    ha="center",
    va="center",
    bbox=box_blue,
    fontsize=9,
    weight="bold",
)

# Decision Diamond / Box
ax.text(
    0.72,
    0.35,
    "Step 4: Strict-Descent Safeguard\nIs Loss(s_jump) < Loss(s_pre)?",
    ha="center",
    va="center",
    bbox=box_gold,
    fontsize=9.5,
    weight="bold",
)

# Branches
ax.text(
    0.40,
    0.15,
    "ACCEPT JUMP\nReset History & Accelerate\n(Loss cuts 40%, 60/60 wins)",
    ha="center",
    va="center",
    bbox=box_green,
    fontsize=9,
    weight="bold",
)
ax.text(
    0.92,
    0.15,
    "REJECT JUMP\nRevert to s_pre\n(Zero penalty)",
    ha="center",
    va="center",
    bbox=box_red,
    fontsize=9,
    weight="bold",
)

# Arrows
ax.annotate("", xy=(0.27, 0.75), xytext=(0.23, 0.75), arrowprops=dict(arrowstyle="->", lw=2, color="#0066cc"))
ax.annotate("", xy=(0.58, 0.75), xytext=(0.53, 0.75), arrowprops=dict(arrowstyle="->", lw=2, color="#0066cc"))
ax.annotate("", xy=(0.72, 0.48), xytext=(0.72, 0.63), arrowprops=dict(arrowstyle="->", lw=2, color="#0066cc"))
ax.annotate("", xy=(0.52, 0.15), xytext=(0.63, 0.28), arrowprops=dict(arrowstyle="->", lw=2, color="#009933"))
ax.annotate("YES (Strict Descent)", xy=(0.54, 0.24), fontsize=8.5, color="#009933", weight="bold")
ax.annotate("", xy=(0.85, 0.15), xytext=(0.78, 0.28), arrowprops=dict(arrowstyle="->", lw=2, color="#cc0000"))
ax.annotate("NO (Stall/Noise)", xy=(0.83, 0.24), fontsize=8.5, color="#cc0000", weight="bold")

ax.set_title("Safeguarded Type-II Anderson Acceleration Workflow", fontsize=12, weight="bold", pad=15)
plt.tight_layout()
diag2_path = os.path.join("results", "diagram_anderson_architecture.png")
plt.savefig(diag2_path)
plt.close()
print(f"Generated: {diag2_path}")
