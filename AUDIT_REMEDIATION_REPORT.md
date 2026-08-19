# Audit Remediation Report

Date: 2026-08-17

## External Context Incorporated

- `C:\Users\hp\Downloads\Spatial_Phase_Space_Telemetry_Research_Report.docx` was used as the historical research log.
- `C:\Users\hp\.codex\attachments\61a480f5-228e-42b9-b0c9-c1818d6f211f\pasted-text.txt` was used as the newer conversation/log context.
- The newer logs supersede the repository's older mini-batch script: they describe a four-variant mini-batch closure, while the checked-in script previously implemented only two variants.

## Remedies Applied

- Final-loss comparison parity: `train_baseline` now returns a post-update evaluation, matching `train_with_anderson`.
- Ito/Xue objective correction: snapshot residual mixing now solves `min ||R alpha||^2` subject to `sum(alpha)=1` through the KKT system.
- Experiment failure propagation: `experiments/run_all.py` now uses `subprocess.run(..., check=True)` so failures stop the suite.
- Provenance: `src/provenance.py` writes JSON result files with environment, config, aggregate statistics, and per-seed raw losses.
- Real-network and saddle experiments now write JSON outputs under `results/`.
- Mini-batch experiment now implements four explicit variants with deterministic per-seed batch schedules shared by all methods.
- Activation smoothness now has a stricter same-seed, same-learning-rate ablation in `experiments/test_smoothness_ablation.py`.
- README and Markdown reports now mark older smoothness results as historical tuned-condition results rather than final causal proof.

## Still Not Conclusive

- The activation-smoothness hypothesis remains unconfirmed until the new same-seed, same-learning-rate ablation is run at the desired seed count.
- Existing PDF output is stale because it predates the audit addendum and code fixes.
- Mini-batch conclusions should be regenerated from the corrected harness because the old script reused one shuffled loader sequentially and did not implement all claimed variants.
- Full generalization remains untested: all real-network experiments still use small models and the scikit-learn digits dataset.
- Evaluation still reports training loss unless a train/validation/test split is explicitly added to a future experiment.

## Verification Plan

- Run `python -m compileall src experiments` after edits.
- Run smoke checks with reduced seed counts for the new mini-batch and smoothness scripts.
- For publishable numbers, rerun full experiments and regenerate Markdown/PDF from JSON provenance instead of hand-copying statistics.
