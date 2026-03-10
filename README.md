# UPM Carpet Anomaly Detection (MVTec AD)

## 1. Problem Statement
This project solves anomaly detection on textured industrial surfaces (`carpet` class from MVTec AD).

Goals:
1. Detect if an image is normal or defective.
2. Localize anomalous regions at pixel level.

The anomaly definition is learned only from normal training images (`train/good`).

## 2. Dataset
Expected dataset structure:

- `train/good/*.png`
- `test/<defect_type>/*.png`
- `ground_truth/<defect_type>/*_mask.png`

Example local path used in experiments:
- `C:\Users\PC\Desktop\carpet`

## 3. Methods Implemented
Two methods were implemented and compared in the same codebase.

### 3.1 PatchCore (final selected method)
Pipeline:
1. Extract multi-scale features from pretrained ResNet (`layer2` + `layer3`).
2. Build a memory bank of normal patch embeddings.
3. Compress memory bank using coreset sampling.
4. At inference, compute nearest-neighbor distance for each patch.
5. Upsample patch anomaly map to image size.
6. Apply thresholding + connected-component filtering (`min_region_area`).

### 3.2 PaDiM (baseline)
Pipeline:
1. Extract feature embeddings from pretrained ResNet.
2. Randomly select embedding dimensions.
3. For each spatial location, estimate a Gaussian model (`mean`, `covariance`) on normal data.
4. At inference, compute Mahalanobis distance map.
5. Threshold and post-process similarly.

## 4. Final Experimental Setup
- Device: CPU
- Backbone: `resnet18`
- Image size: `256`
- Dataset: MVTec AD `carpet`

PatchCore final model:
- Checkpoint: `artifacts/pc_I.pt`
- Postprocess: `min_region_area = 64`

PaDiM comparison model:
- Checkpoint: `artifacts/padim_carpet.pt`
- Postprocess: `min_region_area = 64`

## 5. Results (Fair Comparison)

| Method | Image ROC-AUC | Pixel ROC-AUC | Pixel Precision | Pixel Recall | Pixel F1 | Train Time (s) | Prediction Time (s) | Model Tensor Memory (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PatchCore | 0.9944 | 0.9899 | 0.5526 | 0.7589 | 0.6396 | 114.13 | 10.84 | 6.00 |
| PaDiM | 0.9791 | 0.9866 | 0.5145 | 0.7326 | 0.6044 | 47.85 | 8.70 | 39.45 |

Interpretation:
- PatchCore gives better image-level and pixel-level quality.
- PaDiM trains faster, but memory footprint is much larger in this implementation.
- Final choice: **PatchCore** for best defect detection/localization quality and smaller deployment memory.

## 6. Why PaDiM Memory Is Larger Here
In this project:
- PatchCore uses a compressed memory bank (`4096` patches), giving ~`6 MB`.
- PaDiM stores per-location Gaussian statistics with full covariance, giving ~`39 MB` at `embedding_dim=100`.

So PaDiM is not always lighter; it depends on implementation and hyperparameters.

## 7. Reproducibility
### 7.1 Install dependencies
```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r .\requirements.txt
```

### 7.2 Train PatchCore
```powershell
& ".\.venv\Scripts\python.exe" .\train.py --method patchcore --carpet-root "C:\Users\PC\Desktop\carpet" --device cpu --backbone resnet18 --image-size 288 --batch-size 4 --coreset-method kcenter --coreset-ratio 0.2 --coreset-max-candidates 40000 --coreset-max-selected 4096 --coreset-proj-dim 256 --knn-k 1 --map-smooth-kernel 1 --score-topk-ratio 0.005 --calibrate-quantile 99.5 --output-model ".\artifacts\pc_I.pt"
```

### 7.3 Evaluate PatchCore (final)
```powershell
& ".\.venv\Scripts\python.exe" .\evaluate.py --method auto --carpet-root "C:\Users\PC\Desktop\carpet" --model-path ".\artifacts\pc_I.pt" --device cpu --output-dir ".\outputs\eval_patchcore_fair" --threshold-mode best --min-region-area 64 --visualize-top-k 0
```

### 7.4 Train PaDiM
```powershell
& ".\.venv\Scripts\python.exe" .\train.py --method padim --carpet-root "C:\Users\PC\Desktop\carpet" --device cpu --backbone resnet18 --image-size 256 --batch-size 8 --padim-embedding-dim 100 --padim-cov-eps 0.01 --map-smooth-kernel 1 --score-topk-ratio 0.01 --calibrate-quantile 99.5 --output-model ".\artifacts\padim_carpet.pt"
```

### 7.5 Evaluate PaDiM
```powershell
& ".\.venv\Scripts\python.exe" .\evaluate.py --method auto --carpet-root "C:\Users\PC\Desktop\carpet" --model-path ".\artifacts\padim_carpet.pt" --device cpu --output-dir ".\outputs\eval_padim" --threshold-mode best --min-region-area 64 --visualize-top-k 40
```

### 7.6 Generate comparison table (CSV + JSON)
```powershell
& ".\.venv\Scripts\python.exe" .\compare_methods.py --patchcore-model ".\artifacts\pc_I.pt" --patchcore-train-report ".\artifacts\pc_I.train_report.json" --patchcore-metrics ".\outputs\eval_patchcore_fair\metrics.json" --padim-model ".\artifacts\padim_carpet.pt" --padim-train-report ".\artifacts\padim_carpet.train_report.json" --padim-metrics ".\outputs\eval_padim\metrics.json" --output-csv ".\outputs\comparison\patchcore_vs_padim.csv" --output-json ".\outputs\comparison\patchcore_vs_padim.json"
```

## 8. Key Design Decisions
- Use pretrained CNN backbone and no supervised defect labels in training.
- Evaluate both image-level and pixel-level metrics.
- Keep postprocessing explicit (`min_region_area`) and reproducible.
- Compare two industrially relevant unsupervised methods under the same dataset/protocol.

## 9. Assumptions
- All training images are defect-free.
- Ground-truth masks are available only for defective test samples.
- CPU-only environment is acceptable for this assignment.

## 10. Limitations
- Experiments are on one MVTec category (`carpet`) only.
- Thresholds in `best` mode are selected on test scores (optimistic estimate).
- No cross-category validation or statistical confidence intervals.

## 11. What I Would Improve With More Time
1. Add validation protocol without using test-set threshold optimization.
2. Add category-wise experiments across more MVTec classes.
3. Add deployment profiling (peak RAM, throughput, startup latency) with repeated runs.
4. Try larger backbones with a stricter speed-memory budget.

## 12. Project Structure
- `src/data.py`: dataset loaders and transforms
- `src/features.py`: backbone feature extractor
- `src/patchcore.py`: PatchCore implementation
- `src/padim.py`: PaDiM implementation
- `src/model_utils.py`: auto model loading utility
- `train.py`: training entry point
- `evaluate.py`: evaluation + metrics + visualizations
- `infer.py`: single-image inference
- `compare_methods.py`: method comparison report
