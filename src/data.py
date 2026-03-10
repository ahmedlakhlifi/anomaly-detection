from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def build_image_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def build_mask_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )


class CarpetTrainDataset(Dataset):
    def __init__(self, carpet_root: str, image_size: int = 256):
        self.root = Path(carpet_root)
        self.transform = build_image_transform(image_size)

        train_good = self.root / "train" / "good"
        if not train_good.is_dir():
            raise RuntimeError(f"train/good folder not found in: {self.root}")

        self.image_paths = sorted([p for p in train_good.iterdir() if p.is_file() and _is_image_file(p)])
        if not self.image_paths:
            raise RuntimeError(f"No training images found in: {train_good}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        x = Image.open(p).convert("RGB")
        return self.transform(x), str(p)


class CarpetTestDataset(Dataset):
    def __init__(self, carpet_root: str, image_size: int = 256):
        self.root = Path(carpet_root)
        self.img_tf = build_image_transform(image_size)
        self.mask_tf = build_mask_transform(image_size)
        self.image_size = image_size

        self.samples = []
        test_root = self.root / "test"
        gt_root = self.root / "ground_truth"

        if not test_root.is_dir():
            raise RuntimeError(f"test folder not found in: {self.root}")

        for defect_dir in sorted([d for d in test_root.iterdir() if d.is_dir()]):
            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1

            for img_path in sorted([p for p in defect_dir.iterdir() if p.is_file() and _is_image_file(p)]):
                mask_path = None
                if label == 1:
                    mask_path = gt_root / defect_type / f"{img_path.stem}_mask.png"
                    if not mask_path.exists():
                        raise RuntimeError(f"Missing mask: {mask_path}")
                self.samples.append((img_path, label, mask_path))

        if not self.samples:
            raise RuntimeError(f"No test images found in: {test_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.samples[idx]

        x = Image.open(img_path).convert("RGB")
        x = self.img_tf(x)

        if mask_path is None:
            m = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        else:
            mm = Image.open(mask_path).convert("L")
            m = (self.mask_tf(mm) > 0.5).float()

        return x, m, label, str(img_path)