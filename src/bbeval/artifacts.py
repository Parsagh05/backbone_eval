"""Storage for anomaly maps, scores and ground truth (pptx slide 21).

Maps and raw scores are stored so metrics can be recomputed without re-running
inference -- a different metric or aggregation should never require GPU time,
and should never risk running under a silently different configuration.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

from .config import BackboneEvalConfig
from .corruptions import geometric_names


def shard_path(config: BackboneEvalConfig, backbone_name: str, prompt_mode: str,
               dataset: str, category: str, corruption: str, severity: int) -> str:
    directory = os.path.join(config.artifact_dir, backbone_name, prompt_mode,
                             dataset, category)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{corruption}_s{severity}.npz")


def ground_truth_path(config: BackboneEvalConfig, dataset: str, category: str,
                      corruption: str, severity: int) -> str:
    """Geometric warps move the defect, so those masks are stored per setting."""
    directory = os.path.join(config.artifact_dir, "_ground_truth", dataset, category)
    os.makedirs(directory, exist_ok=True)
    stem = (f"{corruption}_s{severity}"
            if corruption in geometric_names(config) else "base")
    return os.path.join(directory, f"{stem}.npz")


def save_ground_truth(config: BackboneEvalConfig, dataset: str, category: str,
                      corruption: str, severity: int, masks: np.ndarray,
                      labels: np.ndarray) -> str:
    path = ground_truth_path(config, dataset, category, corruption, severity)
    if os.path.isfile(path):
        return path
    np.savez_compressed(path, masks=np.packbits(masks, axis=-1),
                        labels=labels.astype(np.uint8),
                        shape=np.array(masks.shape, dtype=np.int32))
    return path


def load_ground_truth(config: BackboneEvalConfig, dataset: str, category: str,
                      corruption: str = "clean",
                      severity: int = 0) -> tuple[np.ndarray, np.ndarray]:
    stored = np.load(ground_truth_path(config, dataset, category, corruption, severity))
    shape = tuple(int(value) for value in stored["shape"])
    masks = np.unpackbits(stored["masks"], axis=-1, count=shape[-1])
    return masks.reshape(shape).astype(np.uint8), stored["labels"].astype(np.int64)


def save_shard(config: BackboneEvalConfig, backbone, prompt_mode: str, dataset: str,
               category: str, corruption: str, severity: int, scores: np.ndarray,
               maps: np.ndarray, labels: np.ndarray) -> str:
    path = shard_path(config, backbone.name, prompt_mode, dataset, category,
                      corruption, severity)
    meta = {
        "prompt_mode": prompt_mode, "dataset": dataset, "category": category,
        "corruption": corruption, "severity": int(severity),
        "input_size": config.input_size, "map_res": config.map_res,
        "seed": config.seed, "config_id": config.fingerprint(),
        **backbone.describe(),
    }
    np.savez_compressed(path, scores=scores, maps=maps, labels=labels,
                        meta=json.dumps(meta))
    return path


def load_shard(config: BackboneEvalConfig, backbone_name: str, prompt_mode: str,
               dataset: str, category: str, corruption: str,
               severity: int) -> dict | None:
    path = shard_path(config, backbone_name, prompt_mode, dataset, category,
                      corruption, severity)
    if not os.path.isfile(path):
        return None
    stored = np.load(path, allow_pickle=False)
    return {"scores": stored["scores"].astype(np.float64),
            "maps": stored["maps"].astype(np.float32),
            "labels": stored["labels"].astype(np.int64),
            "meta": json.loads(str(stored["meta"]))}


def shard_is_done(config: BackboneEvalConfig, backbone_name: str, dataset: str,
                  category: str, corruption: str, severity: int) -> bool:
    return all(os.path.isfile(shard_path(config, backbone_name, mode, dataset,
                                         category, corruption, severity))
               for mode in config.prompt_modes)


def archive_output(config: BackboneEvalConfig) -> str:
    """Collect every artefact into one ZIP, for a single-file download.

    Written *beside* `output_root`, never inside it: an archive being created
    within its own source tree tries to add itself while still being written.
    """
    output_root = os.path.abspath(config.output_root)
    parent = os.path.dirname(output_root.rstrip(os.sep)) or "."
    os.makedirs(parent, exist_ok=True)
    base = os.path.join(parent, f"backbone_eval_{config.fingerprint()}")
    return shutil.make_archive(base, "zip", root_dir=output_root)


def write_run_manifest(config: BackboneEvalConfig, descriptions: dict) -> str:
    """Record the resolved configuration and every per-backbone choice.

    This is the parity table the comparison rests on: each backbone gets its own
    preprocessing, pooling and readout, and all of it is written down.
    """
    os.makedirs(config.output_root, exist_ok=True)
    path = os.path.join(config.output_root, f"run_manifest_{config.fingerprint()}.json")
    payload = {"config_id": config.fingerprint(), "config": config.to_dict(),
               "backbones": descriptions}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    return path
