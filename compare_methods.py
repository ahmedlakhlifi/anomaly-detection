import argparse
import csv
import json
import math
from pathlib import Path

import torch


def mb(x_bytes: float) -> float:
    return float(x_bytes) / (1024.0 * 1024.0)


def read_json_object(path: Path, label: str) -> dict:
    if not path.exists():
        raise ValueError(f"Missing {label}: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {label} ({path}): {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {label} ({path}), got {type(data).__name__}")
    return data


def require_key(d: dict, key: str, label: str):
    if key not in d:
        raise ValueError(f"Missing required key '{key}' in {label}")
    return d[key]


def require_str(d: dict, key: str, label: str) -> str:
    v = require_key(d, key, label)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"Invalid '{key}' in {label}: expected non-empty string")
    return v


def require_bool(d: dict, key: str, label: str) -> bool:
    v = require_key(d, key, label)
    if not isinstance(v, bool):
        raise ValueError(f"Invalid '{key}' in {label}: expected boolean")
    return v


def require_float(d: dict, key: str, label: str, allow_nan: bool = False) -> float:
    raw = require_key(d, key, label)
    try:
        v = float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid '{key}' in {label}: expected numeric, got {raw!r}") from e

    if math.isinf(v):
        raise ValueError(f"Invalid '{key}' in {label}: infinite values are not allowed")
    if not allow_nan and math.isnan(v):
        raise ValueError(f"Invalid '{key}' in {label}: NaN is not allowed")
    return v


def estimate_runtime_sec(report: dict) -> float:
    fit = require_float(report, "fit_time_sec", "train report")
    cal = require_float(report, "calibration_time_sec", "train report")
    return fit + cal


def estimate_model_memory_mb(model_path: Path) -> tuple[float, float]:
    if not model_path.exists():
        raise ValueError(f"Missing model checkpoint: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")

    tensor_bytes = 0
    if isinstance(ckpt, dict):
        if "memory_bank" in ckpt:
            t = ckpt["memory_bank"]
            tensor_bytes = int(t.numel() * t.element_size())
        elif "mean" in ckpt and "inv_cov" in ckpt:
            t1 = ckpt["mean"]
            t2 = ckpt["inv_cov"]
            tensor_bytes = int(t1.numel() * t1.element_size() + t2.numel() * t2.element_size())

    file_bytes = int(model_path.stat().st_size)
    return mb(tensor_bytes), mb(file_bytes)


def normalize_model_path(path: Path) -> str:
    parts = list(path.parts)
    for i, part in enumerate(parts):
        if part.lower() == "artifacts":
            return "/".join(parts[i:])
    return path.name


def build_row(name: str, model_path: str, train_report_path: str, metrics_path: str) -> dict:
    model_p = Path(model_path)
    report_p = Path(train_report_path)
    metrics_p = Path(metrics_path)

    report = read_json_object(report_p, "train report")
    metrics = read_json_object(metrics_p, "metrics")

    raw_method = report.get("method", metrics.get("method_used"))
    if not isinstance(raw_method, str) or not raw_method.strip():
        raise ValueError(
            "Missing required method information: expected train report key 'method' "
            "or metrics key 'method_used'"
        )
    method = raw_method
    backbone = require_str(report, "backbone", "train report")

    threshold_mode = require_str(metrics, "threshold_mode_requested", "metrics")
    oracle_thresholds = require_bool(metrics, "uses_test_score_oracle_thresholds", "metrics")

    prediction_time_sec = require_float(metrics, "prediction_time_sec", "metrics")
    eval_time_sec = require_float(metrics, "evaluation_time_sec", "metrics")

    image_roc_auc = require_float(metrics, "image_roc_auc", "metrics")
    pixel_roc_auc = require_float(metrics, "pixel_roc_auc", "metrics")
    image_f1 = require_float(metrics, "image_f1_at_used", "metrics")
    pixel_precision = require_float(metrics, "pixel_precision_at_used", "metrics")
    pixel_recall = require_float(metrics, "pixel_recall_at_used", "metrics")
    pixel_f1 = require_float(metrics, "pixel_f1_at_used", "metrics")

    tensor_mem_mb, file_mem_mb = estimate_model_memory_mb(model_p)

    return {
        "name": name,
        "method": method,
        "backbone": backbone,
        "threshold_mode": threshold_mode,
        "oracle_thresholds": oracle_thresholds,
        "train_time_sec": estimate_runtime_sec(report),
        "prediction_time_sec": prediction_time_sec,
        "eval_time_sec": eval_time_sec,
        "image_roc_auc": image_roc_auc,
        "pixel_roc_auc": pixel_roc_auc,
        "image_f1": image_f1,
        "pixel_precision": pixel_precision,
        "pixel_recall": pixel_recall,
        "pixel_f1": pixel_f1,
        "model_tensor_mem_mb": tensor_mem_mb,
        "model_file_mb": file_mem_mb,
        "model_path": normalize_model_path(model_p),
    }


def format_table(rows: list[dict]) -> str:
    headers = [
        "name",
        "method",
        "backbone",
        "thr_mode",
        "oracle_thr",
        "img_auc",
        "pix_auc",
        "pix_prec",
        "pix_rec",
        "pix_f1",
        "train_s",
        "pred_s",
        "tensor_mb",
    ]

    lines = []
    lines.append(" | ".join(headers))
    lines.append(" | ".join(["---"] * len(headers)))

    for r in rows:
        lines.append(
            " | ".join(
                [
                    str(r["name"]),
                    str(r["method"]),
                    str(r["backbone"]),
                    str(r["threshold_mode"]),
                    str(r["oracle_thresholds"]),
                    f"{r['image_roc_auc']:.4f}",
                    f"{r['pixel_roc_auc']:.4f}",
                    f"{r['pixel_precision']:.4f}",
                    f"{r['pixel_recall']:.4f}",
                    f"{r['pixel_f1']:.4f}",
                    f"{r['train_time_sec']:.2f}",
                    f"{r['prediction_time_sec']:.2f}",
                    f"{r['model_tensor_mem_mb']:.2f}",
                ]
            )
        )

    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare PatchCore and PaDiM metrics/runtimes/memory from already-generated reports."
    )
    p.add_argument("--patchcore-model", type=str, required=True)
    p.add_argument("--patchcore-train-report", type=str, required=True)
    p.add_argument("--patchcore-metrics", type=str, required=True)

    p.add_argument("--padim-model", type=str, required=True)
    p.add_argument("--padim-train-report", type=str, required=True)
    p.add_argument("--padim-metrics", type=str, required=True)

    p.add_argument("--output-csv", type=str, default=None)
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    rows = [
        build_row(
            name="patchcore",
            model_path=args.patchcore_model,
            train_report_path=args.patchcore_train_report,
            metrics_path=args.patchcore_metrics,
        ),
        build_row(
            name="padim",
            model_path=args.padim_model,
            train_report_path=args.padim_train_report,
            metrics_path=args.padim_metrics,
        ),
    ]

    print(format_table(rows))

    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

