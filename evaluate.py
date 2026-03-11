import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.data import CarpetTestDataset
from src.model_utils import load_model


def safe_auc(y_true, y_score):
    return float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size > 1 else float("nan")


def best_f1_threshold(y_true, y_score):
    p, r, t = precision_recall_curve(y_true, y_score)
    if t.size == 0:
        return float("nan"), float("nan")
    f1 = (2 * p[:-1] * r[:-1]) / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(f1[i]), float(t[i])


def normalize_uint8(x):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn:
        return np.zeros_like(x, dtype=np.uint8)
    return ((x - mn) / (mx - mn) * 255.0).astype(np.uint8)


def overlay_mask(img, mask, color_bgr):
    out = img.copy()
    idx = mask > 0
    if np.any(idx):
        out[idx] = (0.6 * out[idx] + 0.4 * np.array(color_bgr)).astype(np.uint8)
    return out


def postprocess_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(np.uint8)

    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == i] = 1
    return out


def save_viz(img_path, gt_mask, an_map, pred_mask, out_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return
    h, w = img.shape[:2]
    gt = cv2.resize((gt_mask > 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    pred = cv2.resize((pred_mask > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    amap = cv2.resize(an_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    heat = cv2.applyColorMap(normalize_uint8(amap), cv2.COLORMAP_JET)
    heat_overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0.0)
    gt_overlay = overlay_mask(img, gt, (0, 0, 255))
    pred_overlay = overlay_mask(img, pred, (0, 255, 0))

    canvas = np.concatenate([img, gt_overlay, heat_overlay, pred_overlay], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate PatchCore/PaDiM on MVTec carpet. "
            "Use --threshold-mode calibrated for deployment-fair operating points. "
            "Use --threshold-mode best only as oracle upper-bound analysis."
        )
    )
    p.add_argument("--method", type=str, default="auto", choices=["auto", "patchcore", "padim"], help="Model family to load. auto = detect from checkpoint.")

    p.add_argument("--carpet-root", type=str, required=True, help="Path to MVTec carpet root folder.")
    p.add_argument("--model-path", type=str, required=True, help="Path to saved model checkpoint.")
    p.add_argument("--image-size", type=int, default=None, help="Defaults to model train size if available.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument(
        "--threshold-mode",
        type=str,
        default="calibrated",
        choices=["calibrated", "best"],
        help="calibrated = thresholds from training calibration; best = oracle threshold from test score distribution.",
    )
    p.add_argument("--image-threshold", type=float, default=None, help="Manual image-level threshold override.")
    p.add_argument("--pixel-threshold", type=float, default=None, help="Manual pixel-level threshold override.")

    p.add_argument("--min-region-area", type=int, default=0, help="Connected-component area filtering after pixel thresholding.")

    p.add_argument("--output-dir", type=str, default="outputs/eval")
    p.add_argument("--visualize-top-k", type=int, default=20, help="Save top-K highest image-score visualizations.")
    p.add_argument(
        "--visualize-failures-k",
        type=int,
        default=0,
        help="Save up to K false positives and K false negatives as separate visual panels.",
    )
    return p.parse_args()


def main():
    total_t0 = time.time()
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, used_method = load_model(args.model_path, device=args.device, method=args.method)

    if args.image_size is not None:
        image_size = args.image_size
    elif model.train_image_size is not None:
        image_size = int(model.train_image_size[0])
    else:
        image_size = 256

    test_ds = CarpetTestDataset(args.carpet_root, image_size=image_size)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    pred_t0 = time.time()
    anomaly_maps, image_scores, masks, labels, paths = model.predict_loader(test_loader)
    pred_time_sec = time.time() - pred_t0

    labels = labels.astype(np.int64)
    masks_bin = (masks > 0.5).astype(np.uint8)

    # Best thresholds (oracle analysis)
    image_best_f1, image_best_thr = best_f1_threshold(labels, image_scores)
    pix_gt = masks_bin.reshape(-1)
    pix_score = anomaly_maps.reshape(-1)
    pixel_best_f1, pixel_best_thr = best_f1_threshold(pix_gt, pix_score)

    # Selected thresholds for actual predictions
    if args.image_threshold is not None:
        image_thr = float(args.image_threshold)
        image_thr_source = "manual"
    elif args.threshold_mode == "calibrated" and model.image_threshold is not None:
        image_thr = float(model.image_threshold)
        image_thr_source = "calibrated"
    else:
        image_thr = float(image_best_thr)
        image_thr_source = "best"

    if args.pixel_threshold is not None:
        pixel_thr = float(args.pixel_threshold)
        pixel_thr_source = "manual"
    elif args.threshold_mode == "calibrated" and model.pixel_threshold is not None:
        pixel_thr = float(model.pixel_threshold)
        pixel_thr_source = "calibrated"
    else:
        pixel_thr = float(pixel_best_thr)
        pixel_thr_source = "best"

    uses_test_score_oracle_thresholds = bool(
        (image_thr_source == "best") or (pixel_thr_source == "best")
    )

    pred_labels = (image_scores >= image_thr).astype(np.int64)

    pred_masks = (anomaly_maps >= pixel_thr).astype(np.uint8)
    if args.min_region_area > 0:
        for i in range(pred_masks.shape[0]):
            pred_masks[i] = postprocess_mask(pred_masks[i], min_area=args.min_region_area)

    pixel_pred_flat = pred_masks.reshape(-1)

    # Core metrics
    image_auc = safe_auc(labels, image_scores)
    image_ap = float(average_precision_score(labels, image_scores))
    pixel_auc = safe_auc(pix_gt, pix_score)
    pixel_ap = float(average_precision_score(pix_gt, pix_score))

    fp_idx = np.where((labels == 0) & (pred_labels == 1))[0]
    fn_idx = np.where((labels == 1) & (pred_labels == 0))[0]

    metrics = {
        "method_requested": args.method,
        "method_used": used_method,
        "num_test_images": int(len(labels)),
        "num_anomalous_images": int(labels.sum()),
        "num_false_positives": int(fp_idx.size),
        "num_false_negatives": int(fn_idx.size),
        "image_roc_auc": image_auc,
        "image_average_precision": image_ap,
        "image_best_f1": image_best_f1,
        "image_best_threshold": float(image_best_thr),
        "pixel_roc_auc": pixel_auc,
        "pixel_average_precision": pixel_ap,
        "pixel_best_f1": pixel_best_f1,
        "pixel_best_threshold": float(pixel_best_thr),
        "threshold_mode_requested": args.threshold_mode,
        "image_threshold_used": float(image_thr),
        "image_threshold_source": image_thr_source,
        "pixel_threshold_used": float(pixel_thr),
        "pixel_threshold_source": pixel_thr_source,
        "uses_test_score_oracle_thresholds": uses_test_score_oracle_thresholds,
        "image_precision_at_used": float(precision_score(labels, pred_labels, zero_division=0)),
        "image_recall_at_used": float(recall_score(labels, pred_labels, zero_division=0)),
        "image_f1_at_used": float(f1_score(labels, pred_labels, zero_division=0)),
        "pixel_precision_at_used": float(precision_score(pix_gt, pixel_pred_flat, zero_division=0)),
        "pixel_recall_at_used": float(recall_score(pix_gt, pixel_pred_flat, zero_division=0)),
        "pixel_f1_at_used": float(f1_score(pix_gt, pixel_pred_flat, zero_division=0)),
        "min_region_area": int(args.min_region_area),
        "prediction_time_sec": float(pred_time_sec),
        "evaluation_time_sec": float(time.time() - total_t0),
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "score", "pred_label"])
        for pth, y, s, yp in zip(paths, labels, image_scores, pred_labels):
            w.writerow([pth, int(y), float(s), int(yp)])

    with (out_dir / "failure_cases.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "pred_label", "score", "error_type"])
        for pth, y, yp, s in zip(paths, labels, pred_labels, image_scores):
            if int(y) != int(yp):
                if int(y) == 0 and int(yp) == 1:
                    err = "FP"
                elif int(y) == 1 and int(yp) == 0:
                    err = "FN"
                else:
                    err = "UNKNOWN"
                w.writerow([pth, int(y), int(yp), float(s), err])

    k = min(args.visualize_top_k, len(paths))
    top_idx = np.argsort(-image_scores)[:k]
    viz_dir = out_dir / "visualizations"
    for rank, i in enumerate(top_idx, start=1):
        name = f"{rank:02d}_{Path(paths[i]).stem}_score_{image_scores[i]:.4f}.png"
        save_viz(paths[i], masks[i], anomaly_maps[i], pred_masks[i], viz_dir / name)

    if args.visualize_failures_k > 0:
        fail_dir = out_dir / "visualizations_failures"

        if fp_idx.size > 0:
            fp_sorted = fp_idx[np.argsort(-image_scores[fp_idx])]
            for rank, i in enumerate(fp_sorted[: args.visualize_failures_k], start=1):
                name = f"FP_{rank:02d}_{Path(paths[i]).stem}_score_{image_scores[i]:.4f}.png"
                save_viz(paths[i], masks[i], anomaly_maps[i], pred_masks[i], fail_dir / name)

        if fn_idx.size > 0:
            fn_sorted = fn_idx[np.argsort(image_scores[fn_idx])]
            for rank, i in enumerate(fn_sorted[: args.visualize_failures_k], start=1):
                name = f"FN_{rank:02d}_{Path(paths[i]).stem}_score_{image_scores[i]:.4f}.png"
                save_viz(paths[i], masks[i], anomaly_maps[i], pred_masks[i], fail_dir / name)

    if uses_test_score_oracle_thresholds:
        print("warning=oracle_thresholding_used(best mode or fallback to best threshold)")

    print(json.dumps(metrics, indent=2))
    print(f"saved_dir={out_dir.resolve()}")


if __name__ == "__main__":
    main()
