from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .features import ResNetFeatureExtractor


class PatchCore:
    def __init__(
        self,
        device: str = "cpu",
        backbone: str = "resnet18",
        coreset_sampling_ratio: float = 0.01,
        coreset_method: str = "kcenter",
        coreset_max_candidates: int = 20000,
        coreset_max_selected: int = 512,
        coreset_projection_dim: int = 128,
        coreset_seed: int = 42,
        knn_k: int = 1,
        dist_chunk_size: int = 4096,
        map_smooth_kernel: int = 7,
        score_topk_ratio: float = 0.01,
    ):
        if coreset_method not in {"random", "kcenter"}:
            raise ValueError("coreset_method must be one of: random, kcenter")

        self.device = torch.device(device)
        self.backbone = backbone
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.coreset_method = coreset_method
        self.coreset_max_candidates = int(max(1, coreset_max_candidates))
        self.coreset_max_selected = int(max(1, coreset_max_selected))
        self.coreset_projection_dim = int(max(1, coreset_projection_dim))
        self.coreset_seed = int(coreset_seed)
        self.knn_k = int(max(1, knn_k))
        self.dist_chunk_size = int(max(1, dist_chunk_size))

        kernel = int(max(1, map_smooth_kernel))
        if kernel % 2 == 0:
            kernel += 1
        self.map_smooth_kernel = kernel

        self.score_topk_ratio = float(max(0.0, score_topk_ratio))

        self.extractor = ResNetFeatureExtractor(backbone=backbone, pretrained=True).to(self.device).eval()
        self.memory_bank: Optional[torch.Tensor] = None  # [N, C]

        self.image_threshold: Optional[float] = None
        self.pixel_threshold: Optional[float] = None

        self.train_image_size: Optional[tuple[int, int]] = None
        self.raw_patch_count: Optional[int] = None
        self.candidate_patch_count: Optional[int] = None
        self.memory_bank_size: Optional[int] = None

    @torch.no_grad()
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        f2, f3 = self.extractor(x)
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        emb = torch.cat([f2, f3_up], dim=1)
        return emb

    @staticmethod
    def _embedding_to_patches(emb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = emb.shape
        return emb.permute(0, 2, 3, 1).reshape(b * h * w, c)

    def _rng(self) -> torch.Generator:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.coreset_seed)
        return g

    def _project_for_coreset(self, patches: torch.Tensor) -> torch.Tensor:
        dim = int(patches.shape[1])
        target = min(self.coreset_projection_dim, dim)
        if target >= dim:
            return patches

        g = self._rng()
        proj = torch.randn(dim, target, generator=g, dtype=patches.dtype)
        proj = F.normalize(proj, dim=0)
        return patches @ proj

    def _kcenter_greedy_indices(self, feats: torch.Tensor, n_select: int) -> torch.Tensor:
        n = int(feats.shape[0])
        if n_select >= n:
            return torch.arange(n, dtype=torch.long)

        feats = F.normalize(feats.float(), p=2, dim=1)
        g = self._rng()

        selected = torch.empty(n_select, dtype=torch.long)
        first = int(torch.randint(0, n, (1,), generator=g).item())
        selected[0] = first

        min_dist = torch.cdist(feats, feats[first : first + 1]).squeeze(1)
        for i in range(1, n_select):
            idx = int(torch.argmax(min_dist).item())
            selected[i] = idx
            new_dist = torch.cdist(feats, feats[idx : idx + 1]).squeeze(1)
            min_dist = torch.minimum(min_dist, new_dist)

        return selected

    def _sample_coreset(self, patches: torch.Tensor) -> torch.Tensor:
        n_total = int(patches.shape[0])
        target_keep = max(1, int(n_total * self.coreset_sampling_ratio))

        g = self._rng()

        if self.coreset_method == "random":
            keep = min(target_keep, n_total)
            idx = torch.randperm(n_total, generator=g)[:keep]
            self.raw_patch_count = n_total
            self.candidate_patch_count = n_total
            self.memory_bank_size = keep
            return patches[idx]

        if n_total > self.coreset_max_candidates:
            candidate_idx = torch.randperm(n_total, generator=g)[: self.coreset_max_candidates]
            candidate = patches[candidate_idx]
        else:
            candidate = patches

        n_candidate = int(candidate.shape[0])
        keep = min(target_keep, n_candidate, self.coreset_max_selected)

        candidate_proj = self._project_for_coreset(candidate)
        selected_local = self._kcenter_greedy_indices(candidate_proj, keep)

        self.raw_patch_count = n_total
        self.candidate_patch_count = n_candidate
        self.memory_bank_size = keep
        return candidate[selected_local]

    @torch.no_grad()
    def fit(self, train_loader: DataLoader, max_train_batches: Optional[int] = None) -> None:
        all_patches = []

        for i, batch in enumerate(train_loader):
            images = batch[0].to(self.device, non_blocking=True)
            if self.train_image_size is None:
                self.train_image_size = (int(images.shape[-2]), int(images.shape[-1]))

            emb = self._embed(images)
            patches = self._embedding_to_patches(emb)
            patches = F.normalize(patches, p=2, dim=1)
            all_patches.append(patches.cpu())

            if max_train_batches is not None and (i + 1) >= max_train_batches:
                break

        if not all_patches:
            raise RuntimeError("No patches extracted from training data.")

        patches = torch.cat(all_patches, dim=0)
        self.memory_bank = self._sample_coreset(patches).to(self.device)

    @torch.no_grad()
    def _nearest_distance(self, query: torch.Tensor) -> torch.Tensor:
        if self.memory_bank is None:
            raise RuntimeError("Memory bank is empty. Run fit() first.")

        query = F.normalize(query, p=2, dim=1).to(self.device)
        bank = self.memory_bank

        out = []
        for start in range(0, query.shape[0], self.dist_chunk_size):
            q = query[start : start + self.dist_chunk_size]
            dists = torch.cdist(q, bank)
            k = min(self.knn_k, dists.shape[1])
            vals, _ = torch.topk(dists, k=k, largest=False, dim=1)
            out.append(vals.mean(dim=1))
        return torch.cat(out, dim=0)

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
        images = images.to(self.device, non_blocking=True)
        b, _, in_h, in_w = images.shape

        emb = self._embed(images)
        _, _, h, w = emb.shape
        patches = self._embedding_to_patches(emb)

        patch_scores = self._nearest_distance(patches)
        small_map = patch_scores.view(b, h, w)

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
            g = self._rng()
            idx = torch.randperm(pixel_scores_t.numel(), generator=g)[: int(max_pixel_samples)]
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
        if self.memory_bank is None:
            raise RuntimeError("Nothing to save. Run fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "method": "patchcore",
                "memory_bank": self.memory_bank.cpu(),
                "backbone": self.backbone,
                "coreset_sampling_ratio": self.coreset_sampling_ratio,
                "coreset_method": self.coreset_method,
                "coreset_max_candidates": self.coreset_max_candidates,
                "coreset_max_selected": self.coreset_max_selected,
                "coreset_projection_dim": self.coreset_projection_dim,
                "coreset_seed": self.coreset_seed,
                "knn_k": self.knn_k,
                "dist_chunk_size": self.dist_chunk_size,
                "map_smooth_kernel": self.map_smooth_kernel,
                "score_topk_ratio": self.score_topk_ratio,
                "image_threshold": self.image_threshold,
                "pixel_threshold": self.pixel_threshold,
                "train_image_size": self.train_image_size,
                "raw_patch_count": self.raw_patch_count,
                "candidate_patch_count": self.candidate_patch_count,
                "memory_bank_size": self.memory_bank_size,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "PatchCore":
        ckpt = torch.load(path, map_location="cpu")
        model = cls(
            device=device,
            backbone=ckpt.get("backbone", "resnet18"),
            coreset_sampling_ratio=ckpt.get("coreset_sampling_ratio", 0.01),
            coreset_method=ckpt.get("coreset_method", "random"),
            coreset_max_candidates=ckpt.get("coreset_max_candidates", 20000),
            coreset_max_selected=ckpt.get("coreset_max_selected", 512),
            coreset_projection_dim=ckpt.get("coreset_projection_dim", 128),
            coreset_seed=ckpt.get("coreset_seed", 42),
            knn_k=ckpt.get("knn_k", 1),
            dist_chunk_size=ckpt.get("dist_chunk_size", 4096),
            map_smooth_kernel=ckpt.get("map_smooth_kernel", 7),
            score_topk_ratio=ckpt.get("score_topk_ratio", 0.01),
        )
        model.memory_bank = ckpt["memory_bank"].to(model.device)
        model.image_threshold = ckpt.get("image_threshold", None)
        model.pixel_threshold = ckpt.get("pixel_threshold", None)

        train_image_size = ckpt.get("train_image_size", None)
        if train_image_size is not None:
            model.train_image_size = (int(train_image_size[0]), int(train_image_size[1]))

        model.raw_patch_count = ckpt.get("raw_patch_count", None)
        model.candidate_patch_count = ckpt.get("candidate_patch_count", None)
        model.memory_bank_size = ckpt.get("memory_bank_size", None)
        return model
