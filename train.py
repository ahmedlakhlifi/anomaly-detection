import argparse
import json
import time
from pathlib import Path

from torch.utils.data import DataLoader

from src.data import CarpetTrainDataset
from src.padim import PaDiM
from src.patchcore import PatchCore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, default="patchcore", choices=["patchcore", "padim"])

    p.add_argument("--carpet-root", type=str, required=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)  # Windows-safe
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "wide_resnet50_2"])

    # Shared post-scoring behavior
    p.add_argument("--map-smooth-kernel", type=int, default=7)
    p.add_argument("--score-topk-ratio", type=float, default=0.01)

    # PatchCore args
    p.add_argument("--coreset-ratio", type=float, default=0.01)
    p.add_argument("--coreset-method", type=str, default="kcenter", choices=["random", "kcenter"])
    p.add_argument("--coreset-max-candidates", type=int, default=20000)
    p.add_argument("--coreset-max-selected", type=int, default=512)
    p.add_argument("--coreset-proj-dim", type=int, default=128)
    p.add_argument("--coreset-seed", type=int, default=42)
    p.add_argument("--knn-k", type=int, default=1)
    p.add_argument("--dist-chunk-size", type=int, default=4096)

    # PaDiM args
    p.add_argument("--padim-embedding-dim", type=int, default=100)
    p.add_argument("--padim-cov-eps", type=float, default=0.01)
    p.add_argument("--padim-seed", type=int, default=42)

    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--calibrate-quantile", type=float, default=99.5)
    p.add_argument("--calibrate-max-batches", type=int, default=None)

    p.add_argument("--output-model", type=str, default="artifacts/patchcore_carpet.pt")
    return p.parse_args()


def build_model(args):
    if args.method == "patchcore":
        return PatchCore(
            device=args.device,
            backbone=args.backbone,
            coreset_sampling_ratio=args.coreset_ratio,
            coreset_method=args.coreset_method,
            coreset_max_candidates=args.coreset_max_candidates,
            coreset_max_selected=args.coreset_max_selected,
            coreset_projection_dim=args.coreset_proj_dim,
            coreset_seed=args.coreset_seed,
            knn_k=args.knn_k,
            dist_chunk_size=args.dist_chunk_size,
            map_smooth_kernel=args.map_smooth_kernel,
            score_topk_ratio=args.score_topk_ratio,
        )

    return PaDiM(
        device=args.device,
        backbone=args.backbone,
        embedding_dim=args.padim_embedding_dim,
        seed=args.padim_seed,
        cov_eps=args.padim_cov_eps,
        map_smooth_kernel=args.map_smooth_kernel,
        score_topk_ratio=args.score_topk_ratio,
    )


def main():
    args = parse_args()

    train_ds = CarpetTrainDataset(args.carpet_root, image_size=args.image_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda")),
    )

    model = build_model(args)

    t0 = time.time()
    model.fit(train_loader, max_train_batches=args.max_train_batches)
    fit_time_sec = time.time() - t0

    calibration = None
    if args.calibrate_quantile is not None and args.calibrate_quantile > 0:
        q = float(args.calibrate_quantile) / 100.0
        t1 = time.time()
        calibration = model.calibrate(
            train_loader,
            image_quantile=q,
            pixel_quantile=q,
            max_batches=args.calibrate_max_batches,
        )
        calibration["calibration_time_sec"] = time.time() - t1

    out_path = Path(args.output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)

    train_report = {
        "method": args.method,
        "train_images": len(train_ds),
        "fit_time_sec": fit_time_sec,
        "device": args.device,
        "backbone": args.backbone,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "map_smooth_kernel": args.map_smooth_kernel,
        "score_topk_ratio": args.score_topk_ratio,
        "saved_model": str(out_path.resolve()),
    }

    if args.method == "patchcore":
        train_report.update(
            {
                "coreset_ratio": args.coreset_ratio,
                "coreset_method": args.coreset_method,
                "coreset_max_candidates": args.coreset_max_candidates,
                "coreset_max_selected": args.coreset_max_selected,
                "coreset_proj_dim": args.coreset_proj_dim,
                "knn_k": args.knn_k,
                "dist_chunk_size": args.dist_chunk_size,
                "raw_patch_count": model.raw_patch_count,
                "candidate_patch_count": model.candidate_patch_count,
                "memory_bank_size": model.memory_bank_size,
            }
        )
    else:
        stat_mb = 0.0
        if model.mean is not None and model.inv_cov is not None:
            stat_mb = float((model.mean.numel() + model.inv_cov.numel()) * 4.0 / (1024.0 * 1024.0))

        train_report.update(
            {
                "padim_embedding_dim": args.padim_embedding_dim,
                "padim_cov_eps": args.padim_cov_eps,
                "padim_seed": args.padim_seed,
                "padim_selected_dim": int(model.selected_idx.numel()) if model.selected_idx is not None else None,
                "padim_stats_locations": int(model.mean.shape[0]) if model.mean is not None else None,
                "padim_stats_memory_mb": stat_mb,
            }
        )

    if calibration is not None:
        train_report.update(calibration)

    report_path = out_path.with_suffix(".train_report.json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(train_report, f, indent=2)

    print(json.dumps(train_report, indent=2))


if __name__ == "__main__":
    main()
