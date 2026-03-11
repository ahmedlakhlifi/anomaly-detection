# Project overview
After researching anomaly detection methods for textured surfaces, I decided to implement two approaches: **PatchCore**, which I chose for its strong detection accuracy, and **PaDiM** as a lighter baseline to compare against.

# Task definition
Given only defect-free training images:
1. Predict whether a test image is defective.
2. Localize anomalous regions at pixel level.

Dataset split used in this repo:
- Train: `train/good`
- Test: `test/*`
- Pixel masks: `ground_truth/*`

# Repository structure
- `src/data.py` dataset loaders + transforms
- `src/features.py` backbone feature extraction
- `src/patchcore.py` PatchCore implementation
- `src/padim.py` PaDiM implementation
- `src/model_utils.py` checkpoint auto-loading (`patchcore` / `padim`)
- `train.py` unified training entrypoint
- `evaluate.py` unified evaluation entrypoint (fair + oracle support)
- `infer.py` single-image inference
- `compare_methods.py` metrics/runtime/memory comparison report
- `analyze_failures.py` FP/FN pattern summary
- `results/examples/` curated qualitative examples
- `results/metrics/` curated fair/oracle metrics and analyses

# Methods
## PatchCore
- Multi-scale embedding from pretrained ResNet (`layer2` + upsampled `layer3`)
- Memory bank built from normal patches
- Coreset compression for memory/runtime control
- kNN distance anomaly scoring per patch

## PaDiM
- Embedding extraction from pretrained ResNet
- Random feature subspace selection
- Per-location Gaussian modeling on normal training data
- Mahalanobis distance anomaly scoring

# Implementation details
- Single codebase for both methods (`--method patchcore|padim|auto`)
- Shared evaluation pipeline (same dataset loader, same metric code)
- Evaluation supports:
  - `--threshold-mode calibrated` (fair, no test-threshold leakage)
  - `--threshold-mode best` (oracle upper bound)
- Evaluation exports:
  - `metrics.json`
  - `predictions.csv`
  - `failure_cases.csv`
  - qualitative visual panels

# Experimental setup
- Dataset: MVTec AD `carpet`
- Device: CPU
- Backbone: `resnet18`
- Test images: 117 (89 anomalous)

Model checkpoints:
- PatchCore: `artifacts/pc_I.pt`
- PaDiM: `artifacts/padim_carpet.pt`

# Results
## Main result (fair protocol: calibrated thresholds)
Source: `results/metrics/comparison_fair.csv`

| Method | Thr mode | Oracle thr? | Image AUROC | Pixel AUROC | Image F1 | Pixel Precision | Pixel Recall | Pixel F1 | Train s | Pred s | Tensor MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PatchCore | calibrated | False | 0.9944 | 0.9899 | 0.9778 | 0.2267 | 0.9783 | 0.3681 | 114.13 | 10.26 | 6.00 |
| PaDiM | calibrated | False | 0.9791 | 0.9866 | 0.9297 | 0.2138 | 0.9630 | 0.3499 | 47.85 | 5.66 | 39.45 |

Main takeaway :
- PatchCore has better AUROC and image-level F1.
- PaDiM is faster but much heavier in memory (i was suprised at first ).
- Calibrated thresholds prioritize recall, so pixel precision/F1 are lower than oracle mode.

# Fair evaluation protocol
Main table is intentionally **non-leaky**:
1. Train model on `train/good` only.
2. Calibrate thresholds on training-score quantiles (`train.py --calibrate-quantile 99.5`).
3. Evaluate with `evaluate.py --threshold-mode calibrated`.
4. Do **not** select threshold from test labels/scores for main claims.

Notes:
- `min_region_area=0` is used in fair runs to avoid post-processing hyperparameter tuning on test data.
- AUROC/AP remain threshold-independent.

# Oracle / best-threshold analysis
Source: `results/metrics/comparison_oracle.csv`

| Method | Thr mode | Oracle thr? | Image AUROC | Pixel AUROC | Image F1 | Pixel Precision | Pixel Recall | Pixel F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PatchCore | best | True | 0.9944 | 0.9899 | 0.9832 | 0.5517 | 0.7593 | 0.6391 |
| PaDiM | best | True | 0.9791 | 0.9866 | 0.9659 | 0.5096 | 0.7340 | 0.6016 |

Interpretation:
- This is an **upper bound** because thresholds are selected from test score distribution.
- Reported separately on purpose; not the primary claim.

# Qualitative examples
Each panel is: original | GT mask overlay (red) | anomaly heatmap | predicted mask overlay (green).

## PatchCore examples
**True Positive (strong localization)**
![PatchCore TP](results/examples/patchcore_tp.png)

## PaDiM examples
**True Positive (detected, but noisier map)**
![PaDiM TP](results/examples/padim_tp.png)

# Failure cases
Curated failure visuals (fair calibrated mode):

## PatchCore
**False Positive (normal image predicted defective)**
![PatchCore FP](results/examples/patchcore_fp.png)

**False Negative (defect missed)**
![PatchCore FN](results/examples/patchcore_fn.png)

Summary (`results/metrics/failure_analysis_patchcore_fair.json`):
- FP: 3 (all from `test/good`)
- FN: 1 (`thread`)

## PaDiM
**False Positive example**
![PaDiM FP](results/examples/padim_fp.png)

**False Negative example**
![PaDiM FN](results/examples/padim_fn.png)

Summary (`results/metrics/failure_analysis_padim_fair.json`):
- FP: 10 (all from `test/good`)
- FN: 3 (`color`, `metal_contamination`, `thread`)

# Limitations
- Single category only (`carpet`), not a cross-category generalization claim.
- CPU-only profiling in this repo.
- Calibrated threshold based on training-score quantile is simple and conservative; it can reduce precision.
- No dedicated clean validation split from normal data for operating-point tuning.

# Future improvements
1. Add a non-leaky validation protocol (e.g., synthetic anomaly validation split from train domain).
2. Add cross-category MVTec runs with fixed protocol.
3. Profile peak memory and throughput over repeated runs.
4. Evaluate threshold calibration alternatives beyond fixed quantiles.

# Reproducibility
## Dependencies
All dependencies are version-pinned in `requirements.txt`.

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r .\requirements.txt
```

## Dataset root
```powershell
$env:CARPET_ROOT = ".\data\carpet"
```

## Train
```powershell\n## Train PatchCore\n& ".\.venv\Scripts\python.exe" .\train.py --method patchcore --carpet-root "$env:CARPET_ROOT" --device cpu --backbone resnet18 --image-size 288 --batch-size 4 --coreset-method kcenter --coreset-ratio 0.2 --coreset-max-candidates 40000 --coreset-max-selected 4096 --coreset-proj-dim 256 --knn-k 1 --map-smooth-kernel 1 --score-topk-ratio 0.005 --calibrate-quantile 99.5 --output-model .\artifacts\pc_I.pt
\n## Train PaDiM\n& ".\.venv\Scripts\python.exe" .\train.py --method padim --carpet-root "$env:CARPET_ROOT" --device cpu --backbone resnet18 --image-size 256 --batch-size 8 --padim-embedding-dim 100 --padim-cov-eps 0.01 --map-smooth-kernel 1 --score-topk-ratio 0.01 --calibrate-quantile 99.5 --output-model .\artifacts\padim_carpet.pt
```

## Evaluate (fair calibrated)
```powershell
& ".\.venv\Scripts\python.exe" .\evaluate.py --method auto --carpet-root "$env:CARPET_ROOT" --model-path .\artifacts\pc_I.pt --device cpu --output-dir .\outputs\eval_patchcore_fair_cal --threshold-mode calibrated --min-region-area 0 --visualize-top-k 8 --visualize-failures-k 2

& ".\.venv\Scripts\python.exe" .\evaluate.py --method auto --carpet-root "$env:CARPET_ROOT" --model-path .\artifacts\padim_carpet.pt --device cpu --output-dir .\outputs\eval_padim_fair_cal --threshold-mode calibrated --min-region-area 0 --visualize-top-k 8 --visualize-failures-k 2
```

## Evaluate (oracle upper bound)
```powershell
& ".\.venv\Scripts\python.exe" .\evaluate.py --method auto --carpet-root "$env:CARPET_ROOT" --model-path .\artifacts\pc_I.pt --device cpu --output-dir .\outputs\eval_patchcore_oracle_best --threshold-mode best --min-region-area 0 --visualize-top-k 0

& ".\.venv\Scripts\python.exe" .\evaluate.py --method auto --carpet-root "$env:CARPET_ROOT" --model-path .\artifacts\padim_carpet.pt --device cpu --output-dir .\outputs\eval_padim_oracle_best --threshold-mode best --min-region-area 0 --visualize-top-k 0
```

## Compare methods
```powershell\n## Compare (Fair)\n& ".\.venv\Scripts\python.exe" .\compare_methods.py --patchcore-model .\artifacts\pc_I.pt --patchcore-train-report .\artifacts\pc_I.train_report.json --patchcore-metrics .\outputs\eval_patchcore_fair_cal\metrics.json --padim-model .\artifacts\padim_carpet.pt --padim-train-report .\artifacts\padim_carpet.train_report.json --padim-metrics .\outputs\eval_padim_fair_cal\metrics.json --output-csv .\results\metrics\comparison_fair.csv --output-json .\results\metrics\comparison_fair.json
\n## Compare (Oracle)\n& ".\.venv\Scripts\python.exe" .\compare_methods.py --patchcore-model .\artifacts\pc_I.pt --patchcore-train-report .\artifacts\pc_I.train_report.json --patchcore-metrics .\outputs\eval_patchcore_oracle_best\metrics.json --padim-model .\artifacts\padim_carpet.pt --padim-train-report .\artifacts\padim_carpet.train_report.json --padim-metrics .\outputs\eval_padim_oracle_best\metrics.json --output-csv .\results\metrics\comparison_oracle.csv --output-json .\results\metrics\comparison_oracle.json
```


## Failure analysis
```powershell
& ".\.venv\Scripts\python.exe" .\analyze_failures.py --failure-csv .\outputs\eval_patchcore_fair_cal\failure_cases.csv --output-json .\results\metrics\failure_analysis_patchcore_fair.json

& ".\.venv\Scripts\python.exe" .\analyze_failures.py --failure-csv .\outputs\eval_padim_fair_cal\failure_cases.csv --output-json .\results\metrics\failure_analysis_padim_fair.json
```
