from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .features import ResNetFeatureExtractor


class PaDiM:
    def __init__(
        self,
        device: str = "cpu",
        backbone: str = "resnet18",
        embedding_dim: int = 100,
        seed: int = 42,
        cov_eps: float = 0.01,
        map_smooth_kernel: int = 1,
        score_topk_ratio: float = 0.01,
    ):
        self.device = torch.device(device)
        self.backbone = backbone
        self.embedding_dim = int(max(1, embedding_dim))
        self.seed = int(seed)
        self.cov_eps = float(max(0.0, cov_eps))

        kernel = int(max(1, map_smooth_kernel))
        if kernel % 2 == 0:
            kernel += 1
        self.map_smooth_kernel = kernel

        self.score_topk_ratio = float(max(0.0, score_topk_ratio))

        self.extractor = ResNetFeatureExtractor(backbone=backbone, pretrained=True).to(self.device).eval()

        self.selected_idx: Optional[torch.Tensor] = None
        self.mean: Optional[torch.Tensor] = None  # [L, D]
        self.inv_cov: Optional[torch.Tensor] = None  # [L, D, D]

        self.image_threshold: Optional[float] = None
        self.pixel_threshold: Optional[float] = None
        self.train_image_size: Optional[tuple[int, int]] = None

    def _rng(self) -> torch.Generator:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        return g

    @torch.no_grad()
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        f2, f3 = self.extractor(x)
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        emb = torch.cat([f2, f3_up], dim=1)  # [B, C, H, W]
        return emb

    @torch.no_grad()
    def _project(self, emb: torch.Tensor) -> torch.Tensor:
        if self.selected_idx is None:
            raise RuntimeError("PaDiM not fitted. Missing selected feature indices.")
        return emb[:, self.selected_idx, :, :]

    @torch.no_grad()
    def fit(self, train_loader: DataLoader, max_train_batches: Optional[int] = None) -> None:
        sum_x = None
        sum_xx = None
        count = 0

        for i, batch in enumerate(train_loader):
            images = batch[0].to(self.device, non_blocking=True)
            if self.train_image_size is None:
                self.train_image_size = (int(images.shape[-2]), int(images.shape[-1]))

            emb = self._embed(images)
            b, c, h, w = emb.shape

            if self.selected_idx is None:
                d = min(self.embedding_dim, c)
                idx = torch.randperm(c, generator=self._rng())[:d]
                self.selected_idx = idx

            emb = self._project(emb)
            b, d, h, w = emb.shape
            l = h * w
            x = emb.permute(0, 2, 3, 1).reshape(b, l, d).cpu()  # [B, L, D]

            if sum_x is None:
                sum_x = torch.zeros(l, d, dtype=torch.float32)
                sum_xx = torch.zeros(l, d, d, dtype=torch.float32)

            sum_x += x.sum(dim=0)
            sum_xx += torch.einsum("bld,ble->lde", x, x)
            count += b

            if max_train_batches is not None and (i + 1) >= max_train_batches:
                break

        if count < 2 or sum_x is None or sum_xx is None:
            raise RuntimeError("Not enough training samples to estimate PaDiM Gaussian statistics.")

        mean = sum_x / float(count)  # [L, D]

        outer = torch.einsum("ld,le->lde", mean, mean) * float(count)
        cov = (sum_xx - outer) / float(count - 1)

        eye = torch.eye(cov.shape[-1], dtype=torch.float32).unsqueeze(0).expand_as(cov)
        cov = cov + (self.cov_eps * eye)
        inv_cov = torch.linalg.inv(cov)

        self.mean = mean.to(self.device)
        self.inv_cov = inv_cov.to(self.device)

    @torch.no_grad()
    def _aggregate_image_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        flat = anomaly_map.flatten(1)
        if self.score_topk_ratio <= 0.0:
            return flat.max(dim=1).values

        k = max(1, int(flat.shape[1] * self.score_topk_ratio))
        k = min(k, flat.shape[1])
        return torch.topk(flat, k=k, dim=1).values.mean(dim=1)

    @torch.no_grad()
    def predict(self, images: torch.Tensor):
        if self.mean is None or self.inv_cov is None:
            raise RuntimeError("PaDiM model is not fitted. Run fit() first.")

        images = images.to(self.device, non_blocking=True)
        b, _, in_h, in_w = images.shape

        emb = self._embed(images)
        emb = self._project(emb)
        _, d, h, w = emb.shape
        l = h * w

        x = emb.permute(0, 2, 3, 1).reshape(b, l, d)
        diff = x - self.mean.unsqueeze(0)  # [B, L, D]

        md2 = torch.einsum("bld,lde,ble->bl", diff, self.inv_cov, diff).clamp(min=0.0)
        small_map = torch.sqrt(md2).reshape(b, h, w)

        anomaly_map = F.interpolate(
            small_map.unsqueeze(1),
            size=(in_h, in_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        if self.map_smooth_kernel > 1:
            anomaly_map = F.avg_pool2d(
                anomaly_map.unsqueeze(1),
                kernel_size=self.map_smooth_kernel,
                stride=1,
                padding=self.map_smooth_kernel // 2,
            ).squeeze(1)

        image_scores = self._aggregate_image_score(anomaly_map)
        return anomaly_map.cpu(), image_scores.cpu()

    @torch.no_grad()
    def predict_loader(self, test_loader: DataLoader):
        maps_list, scores_list = [], []
        masks_list, labels_list, paths = [], [], []

        for batch in test_loader:
            images = batch[0]
            masks = batch[1]
            labels = batch[2]
            img_paths = batch[3]

            maps, scores = self.predict(images)

            maps_list.append(maps)
            scores_list.append(scores)
            masks_list.append(masks)
            labels_list.append(labels)
            paths.extend(list(img_paths))

        anomaly_maps = torch.cat(maps_list, dim=0).numpy()
        image_scores = torch.cat(scores_list, dim=0).numpy()
        masks = torch.cat(masks_list, dim=0).squeeze(1).numpy()
        labels = torch.as_tensor(torch.cat(labels_list, dim=0)).numpy()
        return anomaly_maps, image_scores, masks, labels, paths

    @torch.no_grad()
    def calibrate(
        self,
        train_loader: DataLoader,
        image_quantile: float = 0.995,
        pixel_quantile: float = 0.995,
        max_batches: Optional[int] = None,
        max_pixel_samples: int = 2_000_000,
    ) -> dict[str, float]:
        if not 0.0 < image_quantile < 1.0:
            raise ValueError("image_quantile must be in (0,1)")
        if not 0.0 < pixel_quantile < 1.0:
            raise ValueError("pixel_quantile must be in (0,1)")

        image_scores = []
        pixel_scores = []

        for i, batch in enumerate(train_loader):
            images = batch[0]
            maps, scores = self.predict(images)
            image_scores.append(scores)
            pixel_scores.append(maps.reshape(-1))

            if max_batches is not None and (i + 1) >= max_batches:
                break

        if not image_scores or not pixel_scores:
            raise RuntimeError("Calibration failed: empty score lists.")

        image_scores_t = torch.cat(image_scores, dim=0)
        pixel_scores_t = torch.cat(pixel_scores, dim=0)

        if max_pixel_samples is not None and pixel_scores_t.numel() > int(max_pixel_samples):
            idx = torch.randperm(pixel_scores_t.numel(), generator=self._rng())[: int(max_pixel_samples)]
            pixel_scores_t = pixel_scores_t[idx]

        self.image_threshold = float(torch.quantile(image_scores_t, image_quantile).item())
        self.pixel_threshold = float(torch.quantile(pixel_scores_t, pixel_quantile).item())

        return {
            "image_threshold": self.image_threshold,
            "pixel_threshold": self.pixel_threshold,
            "image_quantile": image_quantile,
            "pixel_quantile": pixel_quantile,
            "pixel_samples_used": int(pixel_scores_t.numel()),
        }

    def save(self, path: str | Path) -> None:
        if self.mean is None or self.inv_cov is None or self.selected_idx is None:
            raise RuntimeError("Nothing to save. Run fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "method": "padim",
                "backbone": self.backbone,
                "embedding_dim": self.embedding_dim,
                "seed": self.seed,
                "cov_eps": self.cov_eps,
                "map_smooth_kernel": self.map_smooth_kernel,
                "score_topk_ratio": self.score_topk_ratio,
                "selected_idx": self.selected_idx.cpu(),
                "mean": self.mean.cpu(),
                "inv_cov": self.inv_cov.cpu(),
                "image_threshold": self.image_threshold,
                "pixel_threshold": self.pixel_threshold,
                "train_image_size": self.train_image_size,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "PaDiM":
        ckpt = torch.load(path, map_location="cpu")
        model = cls(
            device=device,
            backbone=ckpt.get("backbone", "resnet18"),
            embedding_dim=ckpt.get("embedding_dim", 100),
            seed=ckpt.get("seed", 42),
            cov_eps=ckpt.get("cov_eps", 0.01),
            map_smooth_kernel=ckpt.get("map_smooth_kernel", 1),
            score_topk_ratio=ckpt.get("score_topk_ratio", 0.01),
        )

        model.selected_idx = ckpt["selected_idx"].long()
        model.mean = ckpt["mean"].to(model.device)
        model.inv_cov = ckpt["inv_cov"].to(model.device)
        model.image_threshold = ckpt.get("image_threshold", None)
        model.pixel_threshold = ckpt.get("pixel_threshold", None)

        train_image_size = ckpt.get("train_image_size", None)
        if train_image_size is not None:
            model.train_image_size = (int(train_image_size[0]), int(train_image_size[1]))

        return model
