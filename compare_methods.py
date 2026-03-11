import argparse
import csv
import json
from pathlib import Path

import torch


def mb(x_bytes: float) -> float:
    return float(x_bytes) / (1024.0 * 1024.0)


def estimate_runtime_sec(report: dict) -> float:
    fit = float(report.get("fit_time_sec", 0.0) or 0.0)
    cal = float(report.get("calibration_time_sec", 0.0) or 0.0)
    return fit + cal


def estimate_model_memory_mb(model_path: Path) -> tuple[float, float]:
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


def build_row(name: str, model_path: str, train_report_path: str, metrics_path: str) -> dict:
    model_p = Path(model_path)
    report_p = Path(train_report_path)
    metrics_p = Path(metrics_path)

    report = json.loads(report_p.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_p.read_text(encoding="utf-8"))

    tensor_mem_mb, file_mem_mb = estimate_model_memory_mb(model_p)

    return {
        "name": name,
        "method": report.get("method", metrics.get("method_used", "unknown")),
        "backbone": report.get("backbone", "unknown"),
        "threshold_mode": metrics.get("threshold_mode_requested", "unknown"),
        "oracle_thresholds": bool(metrics.get("uses_test_score_oracle_thresholds", False)),
        "train_time_sec": estimate_runtime_sec(report),
        "prediction_time_sec": float(metrics.get("prediction_time_sec", 0.0) or 0.0),
        "eval_time_sec": float(metrics.get("evaluation_time_sec", 0.0) or 0.0),
        "image_roc_auc": float(metrics.get("image_roc_auc", 0.0) or 0.0),
        "pixel_roc_auc": float(metrics.get("pixel_roc_auc", 0.0) or 0.0),
        "image_f1": float(metrics.get("image_f1_at_used", 0.0) or 0.0),
        "pixel_precision": float(metrics.get("pixel_precision_at_used", 0.0) or 0.0),
        "pixel_recall": float(metrics.get("pixel_recall_at_used", 0.0) or 0.0),
        "pixel_f1": float(metrics.get("pixel_f1_at_used", 0.0) or 0.0),
        "model_tensor_mem_mb": tensor_mem_mb,
        "model_file_mb": file_mem_mb,
        "model_path": str(model_p.resolve()),
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
