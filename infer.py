import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.data import build_image_transform
from src.model_utils import load_model


def normalize_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn:
        return np.zeros_like(x, dtype=np.uint8)
    return ((x - mn) / (mx - mn) * 255.0).astype(np.uint8)


def overlay_mask(img: np.ndarray, mask: np.ndarray, color_bgr=(0, 255, 0)) -> np.ndarray:
    out = img.copy()
    idx = mask > 0
    if np.any(idx):
        out[idx] = (0.6 * out[idx] + 0.4 * np.array(color_bgr)).astype(np.uint8)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, default="auto", choices=["auto", "patchcore", "padim"])

    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--image-path", type=str, required=True)
    p.add_argument("--image-size", type=int, default=None, help="Defaults to model train size if available.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--threshold", type=float, default=None, help="Optional image-level threshold")
    p.add_argument("--pixel-threshold", type=float, default=None, help="Optional pixel-level threshold")
    p.add_argument("--output-dir", type=str, default="outputs/infer")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, used_method = load_model(args.model_path, device=args.device, method=args.method)

    if args.image_size is not None:
        image_size = int(args.image_size)
    elif model.train_image_size is not None:
        image_size = int(model.train_image_size[0])
    else:
        image_size = 256

    img_path = Path(args.image_path)
    pil = Image.open(img_path).convert("RGB")
    tf = build_image_transform(image_size)
    x = tf(pil).unsqueeze(0)

    an_map_t, score_t = model.predict(x)
    an_map = an_map_t[0].numpy()
    score = float(score_t[0].item())

    if args.threshold is not None:
        image_thr = float(args.threshold)
        image_thr_source = "manual"
    elif model.image_threshold is not None:
        image_thr = float(model.image_threshold)
        image_thr_source = "calibrated"
    else:
        image_thr = 0.6435
        image_thr_source = "fallback"

    if args.pixel_threshold is not None:
        pixel_thr = float(args.pixel_threshold)
        pixel_thr_source = "manual"
    elif model.pixel_threshold is not None:
        pixel_thr = float(model.pixel_threshold)
        pixel_thr_source = "calibrated"
    else:
        pixel_thr = float(np.percentile(an_map, 99.0))
        pixel_thr_source = "fallback_percentile"

    pred_label = int(score >= image_thr)
    pred_mask = (an_map >= pixel_thr).astype(np.uint8)

    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    h, w = img_bgr.shape[:2]
    heat = cv2.applyColorMap(normalize_uint8(cv2.resize(an_map, (w, h))), cv2.COLORMAP_JET)
    heat_overlay = cv2.addWeighted(img_bgr, 0.6, heat, 0.4, 0.0)

    pred_mask_rs = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    pred_overlay = overlay_mask(img_bgr, pred_mask_rs, color_bgr=(0, 255, 0))

    panel = np.concatenate([img_bgr, heat_overlay, pred_overlay], axis=1)
    cv2.imwrite(str(out_dir / f"{img_path.stem}_panel.png"), panel)
    cv2.imwrite(str(out_dir / f"{img_path.stem}_heatmap.png"), heat)
    cv2.imwrite(str(out_dir / f"{img_path.stem}_pred_mask.png"), (pred_mask_rs * 255).astype(np.uint8))

    summary = {
        "method_requested": args.method,
        "method_used": used_method,
        "image_path": str(img_path),
        "image_size": image_size,
        "image_score": score,
        "image_threshold": image_thr,
        "image_threshold_source": image_thr_source,
        "pixel_threshold": pixel_thr,
        "pixel_threshold_source": pixel_thr_source,
        "pred_label": pred_label,
    }

    with (out_dir / f"{img_path.stem}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"saved_dir={out_dir.resolve()}")


if __name__ == "__main__":
    main()
