"""Test-split indexing and the dataset object.

Only the test splits of MVTec and VisA are ever read (pptx slide 21). Roots are
supplied by configuration -- nothing is downloaded here, because both datasets
are mounted read-only on Kaggle.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .config import BackboneEvalConfig
from .corruptions import apply_corruption
from .determinism import dataloader_kwargs, derived_seed

Image.MAX_IMAGE_PIXELS = None

MVTEC_CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
)
VISA_CATEGORIES = (
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
)
ALL_CATEGORIES = {"mvtec": MVTEC_CATEGORIES, "visa": VISA_CATEGORIES}

# Category names carry an underscore or a serial number that reads badly in a prompt.
PROMPT_CLASS_NAMES = {
    "metal_nut": "metal nut", "pipe_fryum": "pipe fryum",
    "pcb1": "printed circuit board", "pcb2": "printed circuit board",
    "pcb3": "printed circuit board", "pcb4": "printed circuit board",
    "macaroni1": "macaroni", "macaroni2": "macaroni",
}


def prompt_class_name(category: str) -> str:
    return PROMPT_CLASS_NAMES.get(category, category.replace("_", " "))


def categories_for(config: BackboneEvalConfig, dataset: str) -> tuple[str, ...]:
    if config.categories and dataset in config.categories:
        return config.categories[dataset]
    return ALL_CATEGORIES[dataset]


def index_mvtec_test(root: str, category: str) -> list[dict]:
    base = os.path.join(root, category)
    records = []
    for defect in sorted(os.listdir(os.path.join(base, "test"))):
        image_dir = os.path.join(base, "test", defect)
        if not os.path.isdir(image_dir):
            continue
        for name in sorted(os.listdir(image_dir)):
            if not name.lower().endswith(".png"):
                continue
            stem = os.path.splitext(name)[0]
            records.append({
                "image": os.path.join(image_dir, name),
                "mask": None if defect == "good" else os.path.join(
                    base, "ground_truth", defect, f"{stem}_mask.png"),
                "label": int(defect != "good"),
            })
    return records


def index_visa_test(root: str, category: str) -> list[dict]:
    """Reads VisA's official one-class split; no directory reorganisation needed."""
    records = []
    with open(os.path.join(root, "split_csv", "1cls.csv"),
              newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["object"] != category or row["split"] != "test":
                continue
            is_anomalous = row["label"].strip().lower() != "normal"
            mask = row["mask"].strip()
            records.append({
                "image": os.path.join(root, row["image"].strip()),
                "mask": os.path.join(root, mask) if (is_anomalous and mask) else None,
                "label": int(is_anomalous),
            })
    return sorted(records, key=lambda record: record["image"])


_INDEXERS = {"mvtec": index_mvtec_test, "visa": index_visa_test}
_INDEX_CACHE: dict[tuple[str, str, str], list[dict]] = {}


def index_test_split(config: BackboneEvalConfig, dataset: str, category: str) -> list[dict]:
    root = config.dataset_roots[dataset]
    key = (root, dataset, category)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]
    records = _INDEXERS[dataset](root, category)
    if not records:
        raise RuntimeError(f"no test images found for {dataset}/{category} under {root}")
    missing = [record for record in records
               if not os.path.isfile(record["image"])
               or (record["mask"] and not os.path.isfile(record["mask"]))]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} missing file(s) for {dataset}/{category}, "
            f"first: {missing[0]}")
    _INDEX_CACHE[key] = records
    return records


def load_mask_full(path: str | None, size: int) -> np.ndarray:
    """Binary mask at `size`, before any downsampling to the metric resolution."""
    if path is None:
        return np.zeros((size, size), dtype=np.uint8)
    mask = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def downsample_mask(mask_uint8: np.ndarray, size: int) -> torch.Tensor:
    raw = torch.from_numpy(mask_uint8.astype(np.float32))[None, None]
    reduced = F.interpolate(raw, size=(size, size), mode="area") > 0.5
    if not reduced.any() and raw.any():
        # Preserve defects too small to survive area-downsampling.
        reduced = F.adaptive_max_pool2d(raw, size) > 0
    return reduced[0, 0].to(torch.uint8)


class AnomalyTestSet(Dataset):
    """One (dataset, category) test split, optionally corrupted."""

    def __init__(self, config: BackboneEvalConfig, dataset: str, category: str,
                 corruption: str = "clean", severity: int = 0,
                 limit: int | None = None) -> None:
        self.config = config
        self.dataset = dataset
        self.category = category
        self.corruption = corruption
        self.severity = severity
        self.records = index_test_split(config, dataset, category)
        if limit is not None:
            self.records = self.records[:limit]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        config = self.config
        record = self.records[index]
        image = Image.open(record["image"]).convert("RGB")
        image = image.resize((config.input_size, config.input_size), Image.BICUBIC)
        array = np.asarray(image, dtype=np.uint8)
        mask = load_mask_full(record["mask"], config.input_size)

        array, mask = apply_corruption(
            config, array, mask, self.corruption, self.severity,
            cache_key=f"{self.dataset}/{self.category}/"
                      f"{os.path.basename(record['image'])}")

        # PIL exposes its buffer read-only and the clean path returns it
        # unchanged, so copy before handing it to torch.
        array = np.array(array, dtype=np.uint8, copy=True, order="C")
        return {
            "image": torch.from_numpy(array).permute(2, 0, 1),      # uint8 [3,H,W]
            "mask": downsample_mask(mask, config.map_res),          # uint8 [M,M]
            "label": record["label"],
            "index": index,
        }


def make_loader(config: BackboneEvalConfig, dataset: str, category: str,
                corruption: str = "clean", severity: int = 0,
                limit: int | None = None) -> DataLoader:
    data = AnomalyTestSet(config, dataset, category, corruption, severity, limit)
    return DataLoader(
        data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
        drop_last=False,
        **dataloader_kwargs(config.seed),
    )


def make_train_loader(config: BackboneEvalConfig, source: str) -> DataLoader:
    """Every category of `source`, shuffled, for fitting the prompt context."""
    parts = [AnomalyTestSet(config, source, category,
                            limit=config.max_train_images_per_category)
             for category in categories_for(config, source)]
    return DataLoader(
        ConcatDataset(parts),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
        drop_last=False,
        **dataloader_kwargs(derived_seed(config.seed, "train", source)),
    )
