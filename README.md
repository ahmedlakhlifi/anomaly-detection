# UPM Carpet Anomaly Detection (MVTec AD)

## 1) Problem Statement
This project addresses unsupervised anomaly detection for textured industrial surfaces using the MVTec AD `carpet` category.

Objectives:
1. Detect whether an image is defective.
2. Localize anomalous regions at pixel level.

The anomaly definition is learned from defect-free training images only (`train/good`).

## 2) Dataset
Expected dataset structure:

- `train/good/*.png`
- `test/<defect_type>/*.png`
- `ground_truth/<defect_type>/*_mask.png`

Set a dataset root path (example PowerShell):

```powershell
$env:CARPET_ROOT = ".\data\carpet"
```

You can download MVTec AD from the official MVTec dataset page and place the extracted `carpet` folder at `$CARPET_ROOT`.

## 3) Methods Implemented
- **PatchCore** (selected final method)
- **PaDiM** (baseline)

Both methods use a pretrained `resnet18` feature backbone in this project.

## 4) Final Setup
- Device: CPU
- Backbone: `resnet18`
- Image size: `256`
- Post-processing: `min_region_area = 64`

Checkpoints:
- PatchCore: `artifacts/pc_I.pt`
- PaDiM: `artifacts/padim_carpet.pt`

## 5) Results (Fair Comparison)

| Method | Image ROC-AUC | Pixel ROC-AUC | Pixel Precision | Pixel Recall | Pixel F1 | Train Time (s) | Prediction Time (s) | Model Tensor Memory (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PatchCore | 0.9944 | 0.9899 | 0.5526 | 0.7589 | 0.6396 | 114.13 | 10.84 | 6.00 |
| PaDiM | 0.9791 | 0.9866 | 0.5145 | 0.7326 | 0.6044 | 47.85 | 8.70 | 39.45 |

`*` Threshold note: reported F1/precision/recall use `--threshold-mode best`, so thresholds are selected from test scores (optimistic).

Interpretation:
- PatchCore gives stronger image-level and pixel-level quality.
- PaDiM is faster to train/infer in this setup.
- PatchCore is much lighter in memory here because of aggressive memory-bank compression.

## 6) Visual Examples
PatchCore examples:

![PatchCore Example 1](results/examples/patchcore_example_1.png)
![PatchCore Example 2](results/examples/patchcore_example_2.png)

PaDiM examples:

![PaDiM Example 1](results/examples/padim_example_1.png)
![PaDiM Example 2](results/examples/padim_example_2.png)

## 7) Reproducibility
Install dependencies:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r .\requirements.txt
```

Train PatchCore:

```powershell
& ".\.venv\Scripts\python.exe" .\train.py --method patchcore --carpet-root "$env:CARPET_ROOT" --device cpu --backbone resnet18 --image-size 288 --batch-size 4 --coreset-method kcenter --coreset-ratio 0.2 --coreset-max-candidates 40000 --coreset-max-selected 4096 --coreset-proj-dim 256 --knn-k 1 --map-smooth-kernel 1 --score-topk-ratio 0.005 --calibrate-quantile 99.5 --output-model ".\artifacts\pc_I.pt"
```

Evaluate PatchCore:

```powershell
& ".\.venv\Scripts\python.exe" .\evaluate.py --method auto --carpet-root "$env:CARPET_ROOT" --model-path ".\artifacts\pc_I.pt" --device cpu --output-dir ".\outputs\eval_patchcore_fair" --threshold-mode best --min-region-area 64 --visualize-top-k 0
```

Train PaDiM:

```powershell
& ".\.venv\Scripts\python.exe" .\train.py --method padim --carpet-root "$env:CARPET_ROOT" --device cpu --backbone resnet18 --image-size 256 --batch-size 8 --padim-embedding-dim 100 --padim-cov-eps 0.01 --map-smooth-kernel 1 --score-topk-ratio 0.01 --calibrate-quantile 99.5 --output-model ".\artifacts\padim_carpet.pt"
```

Evaluate PaDiM:

```powershell
& ".\.venv\Scripts\python.exe" .\evaluate.py --method auto --carpet-root "$env:CARPET_ROOT" --model-path ".\artifacts\padim_carpet.pt" --device cpu --output-dir ".\outputs\eval_padim" --threshold-mode best --min-region-area 64 --visualize-top-k 40
```

Generate comparison report:

```powershell
& ".\.venv\Scripts\python.exe" .\compare_methods.py --patchcore-model ".\artifacts\pc_I.pt" --patchcore-train-report ".\artifacts\pc_I.train_report.json" --patchcore-metrics ".\outputs\eval_patchcore_fair\metrics.json" --padim-model ".\artifacts\padim_carpet.pt" --padim-train-report ".\artifacts\padim_carpet.train_report.json" --padim-metrics ".\outputs\eval_padim\metrics.json" --output-csv ".\outputs\comparison\patchcore_vs_padim.csv" --output-json ".\outputs\comparison\patchcore_vs_padim.json"
```

## 8) Tracked Result Artifacts
Small, repo-friendly result files are included in `results/metrics/`:
- `patchcore_metrics.json`
- `padim_metrics.json`
- `patchcore_vs_padim.csv`
- `patchcore_vs_padim.json`

## 9) Assumptions and Limitations
- Training images are defect-free.
- Experiments are only on MVTec AD `carpet`.
- `best` threshold mode uses test score distribution (optimistic operating-point metrics).

## 10) Project Structure
- `src/data.py`: dataset loaders and transforms
- `src/features.py`: backbone feature extractor
- `src/patchcore.py`: PatchCore implementation
- `src/padim.py`: PaDiM implementation
- `src/model_utils.py`: auto model loading utility
- `train.py`: training entry point
- `evaluate.py`: evaluation + metrics + visualizations
- `infer.py`: single-image inference
- `compare_methods.py`: method comparison utility
