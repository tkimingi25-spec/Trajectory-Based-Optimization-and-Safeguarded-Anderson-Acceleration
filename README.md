# Anderson Acceleration & Optimization Phase-Space Telemetry

A verification-driven, reproducible research codebase investigating trajectory-based training acceleration and geometric telemetry in neural network optimization.

---

## Executive Summary

This repository documents and independently reproduces a comprehensive investigation into whether optimizer trajectory geometry can be visualized, diagnosed, and leveraged to accelerate neural network training.

### Key Scientific Findings

1. **Phase-Space Telemetry (§3):** The intuitive 3D optimization spiral is verified as a **single parameter plotted against its own velocity** $(w_1, v_1, t)$—a classical phase portrait—rather than two independent parameters.
2. **Curvature-Informed Acceleration Fails (§5.1–§5.9):** Six external curvature-informed mechanisms (single/multi-eigenvector saddle escape, NAG comparison, velocity blending, gated rescue) were systematically tested and **all rejected**: they either fell into greedy descent traps or failed to generalize across different saddle landscapes.
3. **Safeguarded Anderson Acceleration Succeeds (§5.10–§5.11):** Type-II Anderson acceleration on concatenated position+velocity states $(w, v)$ provides statistically robust acceleration without computing Hessians or gradient vectors. A **strict-descent safeguard** ($\mathcal{L}_{\text{jump}} < \mathcal{L}_{\text{pre}}$) is mathematically essential to prevent catastrophic noise-overfitting near flat regions.
4. **Activation Smoothness Hypothesis (§5.12–§5.13):** Historical tuned-condition runs showed a strong positive Tanh result and a ReLU null result. The stricter same-seed, same-learning-rate ablation now lives in `experiments/test_smoothness_ablation.py` and should be used for causal smoothness claims.
5. **Mini-Batch SGD Stochasticity Breakdown (Test #2):** Under mini-batch training, stochastic gradient noise is tested against four explicit variants: naive classical Anderson, full-dataset-safeguarded classical Anderson, sparse-history classical Anderson, and Ito/Xue snapshot residual mixing.
6. **Late-Training Horizon Dynamics (Test #3):** Extended horizon analysis ($T \in \{100, 250, 500, 1000\}$ steps) reveals that Anderson acceleration's relative advantage peaks in early-to-mid training and gradually saturates as standard SGD reaches asymptotic refinement.

---

## Directory Structure

```
ANDESSON ACCELERATION/
├── requirements.txt             # Minimal dependencies (torch, numpy, scipy, sklearn, matplotlib)
├── README.md                    # Research overview, theoretical background & instructions
├── src/
│   ├── __init__.py
│   ├── models.py                # TinyMLP, SmallCNN (ReLU), SmallCNNTanh (Smooth), LinearAutoencoder
│   ├── anderson.py              # Classical Type-II & Ito-Xue Anderson extrapolators + strict safeguards
│   ├── training.py              # Baseline, Anderson training loops, seed-matched paired benchmarking
│   ├── diagnostics.py           # Trajectory PCA (Engine A), shifted power iteration, explosion guard
│   ├── landscapes.py            # Monotonic quadratic ravines, Baldi-Hornik strict saddles #1 & #2
│   ├── statistics.py            # Paired t-test, Wilcoxon signed-rank test, reporting formatters
│   └── data.py                  # Sklearn digits loader, TinyMLP & Baldi-Hornik synthetic generators
├── experiments/
│   ├── tier1_ravine.py          # §3: Monotonic quadratic ravine & phase portrait generation
│   ├── tier2_diagnostics.py     # §4: Real TinyMLP trajectory PCA & shifted power iteration
│   ├── test_saddle_escape.py    # §5.1-5.9: Negative results (why curvature mechanisms failed)
│   ├── test_anderson_saddles.py # §5.10: Anderson on Baldi-Hornik Saddles #1 (D=24) & #2 (D=60)
│   ├── test_anderson_real.py    # §5.11-5.13: TinyMLP, ReLU CNN (null) & Tanh CNN (positive)
│   ├── test_smoothness_ablation.py # Same-seed, same-lr activation smoothness ablation
│   ├── test_minibatch.py        # Test #2: Mini-batch SGD failure modes
│   ├── test_late_training.py    # Test #3: Late-training horizon dynamics & saturation
│   └── run_all.py               # Master runner executing all experiments sequentially
└── results/                     # Output figures, JSON logs, and analysis artifacts
```

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/anderson-acceleration-research.git
cd "anderson-acceleration-research"

# Install minimal scientific dependencies
pip install -r requirements.txt
```

---

## Reproducing the Experiments

You can execute individual experiments or run the full test suite:

### 1. Monotonic Ravine & Phase Portraits (§3)
```bash
python experiments/tier1_ravine.py
```
*Validates the L-shaped monotonic trajectory on ill-conditioned quadratics and generates the 3D phase portrait $(w_1, v_1, t)$.*

### 2. Real-Network Trajectory PCA & Saddle Telemetry (§4)
```bash
python experiments/tier2_diagnostics.py
```
*Runs PCA on TinyMLP parameter trajectories ($PC_1=93.44\%$, $PC_2=4.86\%$) and verifies the two-stage shifted power iteration for $\lambda_{\min}$.*

### 3. Baldi-Hornik Strict Saddles (§5.10)
```bash
python experiments/test_anderson_saddles.py
```
*Runs 30-seed matched benchmarks on Saddles #1 ($D=24$) and #2 ($D=60$), confirming generalizable saddle escape via Anderson acceleration.*

### 4. Real-Network Smoothness Isolation (§5.11–§5.13)
```bash
python experiments/test_anderson_real.py
```
*Executes the historical tuned-condition comparison between ReLU CNN and Tanh CNN. For causal activation-smoothness evidence, run `experiments/test_smoothness_ablation.py`.*

### 4b. Same-Seed Smoothness Ablation
```bash
python experiments/test_smoothness_ablation.py
```
*Runs ReLU and Tanh with the same seeds and learning-rate grid to avoid conflating activation smoothness with tuning differences.*

### 5. Mini-Batch SGD Stochasticity Analysis (Test #2)
```bash
python experiments/test_minibatch.py
```
*Evaluates naive classical Anderson, full-dataset-safeguarded classical Anderson, sparse-history classical Anderson, and Ito & Xue snapshot mixing under stochastic gradient noise. The full-dataset safeguard is not a held-out validation safeguard.*

### 6. Late-Training Behavior & Horizon Dynamics (Test #3)
```bash
python experiments/test_late_training.py
```
*Evaluates 60 seeds across horizons $T \in \{100, 250, 500, 1000\}$ to map how Anderson's advantage evolves over deep training.*

### 7. Run Full Suite
```bash
python experiments/run_all.py
```

---

## Statistical Standard

In accordance with strict verification methodology:
- Every comparison uses **matched seeds** (identical parameter initialization and data order).
- Both **parametric paired t-tests** (`scipy.stats.ttest_1samp`) and **non-parametric Wilcoxon signed-rank tests** (`scipy.stats.wilcoxon`) are reported.
- Claims require $n \ge 30\text{--}60$ seeds and $p < 0.05$ on both tests.

---

## Key References

- **Anderson, D. G. (1965).** *Iterative Procedures for Nonlinear Integral Equations.* Journal of the ACM, 12(4), 547–560.
- **Walker, H. F., & Ni, P. (2011).** *Anderson Acceleration for Fixed-Point Iterations.* SIAM Journal on Numerical Analysis, 49(4), 1715–1735.
- **Baldi, P., & Hornik, K. (1989).** *Neural Networks and Principal Component Analysis: Learning from Examples Without Local Minima.* Neural Networks, 2(1), 53–58.
- **Ito, K., & Xue, T. (2025).** *Anderson-type acceleration method for deep neural network optimization.*
- **Goh, G. (2017).** *Why Momentum Really Works.* Distill.
- **Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018).** *Visualizing the Loss Landscape of Neural Nets.* NeurIPS.
