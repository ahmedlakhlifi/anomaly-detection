from __future__ import annotations

from pathlib import Path

import torch

from .padim import PaDiM
from .patchcore import PatchCore


def detect_method_from_checkpoint(path: str | Path) -> str:
    ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict):
        method = ckpt.get("method", None)
        if method in {"patchcore", "padim"}:
            return str(method)
        if "memory_bank" in ckpt:
            return "patchcore"
        if "mean" in ckpt and "inv_cov" in ckpt:
            return "padim"

    raise ValueError(f"Could not detect model method from checkpoint: {path}")


def load_model(path: str | Path, device: str = "cpu", method: str = "auto"):
    selected_method = method
    if selected_method == "auto":
        selected_method = detect_method_from_checkpoint(path)

    if selected_method == "patchcore":
        return PatchCore.load(path, device=device), selected_method
    if selected_method == "padim":
        return PaDiM.load(path, device=device), selected_method

    raise ValueError(f"Unsupported method: {selected_method}")
