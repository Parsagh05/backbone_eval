"""Category-level and dataset-level tables from stored shards (pptx slide 21)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .artifacts import load_ground_truth, load_shard
from .config import BackboneEvalConfig
from .corruptions import corruption_grid, corruption_group_of
from .datasets import categories_for
from .metrics import ALL_METRICS, evaluate_shard, image_metrics


def collect_category_table(config: BackboneEvalConfig,
                           verbose: bool = True) -> pd.DataFrame:
    group_of = corruption_group_of(config)
    rows, missing = [], 0
    for backbone_name in config.backbones:
        for mode in config.prompt_modes:
            for source, evaluate_on in config.protocol:
                for category in categories_for(config, evaluate_on):
                    for corruption, severity in corruption_grid(config):
                        shard = load_shard(config, backbone_name, mode, evaluate_on,
                                           category, corruption, severity)
                        if shard is None:
                            missing += 1
                            continue
                        masks, _ = load_ground_truth(config, evaluate_on, category,
                                                     corruption, severity)
                        metrics = evaluate_shard(
                            masks, shard["maps"], shard["labels"], shard["scores"],
                            max_fpr=config.aupro_fpr_limit,
                            num_thresholds=config.aupro_thresholds,
                            ece_bins=config.ece_bins)
                        rows.append({"backbone": backbone_name, "prompt_mode": mode,
                                     "source": source, "dataset": evaluate_on,
                                     "category": category, "corruption": corruption,
                                     "severity": severity,
                                     "group": group_of.get(corruption, "clean"),
                                     "n_images": len(shard["labels"]),
                                     **metrics})
    if verbose and missing:
        print(f"note: {missing} shard(s) not yet computed and skipped")
    return pd.DataFrame(rows)


def pooled_image_metrics(config: BackboneEvalConfig, backbone_name: str,
                         prompt_mode: str, dataset: str, corruption: str,
                         severity: int) -> dict[str, float]:
    """Image metrics over every image of a dataset at once, not per category."""
    labels, scores = [], []
    for category in categories_for(config, dataset):
        shard = load_shard(config, backbone_name, prompt_mode, dataset, category,
                           corruption, severity)
        if shard is None:
            return {}
        labels.append(shard["labels"])
        scores.append(shard["scores"])
    metrics = image_metrics(np.concatenate(labels), np.concatenate(scores))
    return {f"{key}_pooled": value for key, value in metrics.items()}


def build_dataset_table(config: BackboneEvalConfig,
                        category_table: pd.DataFrame) -> pd.DataFrame:
    """Dataset-level = unweighted mean over categories, as in AnomalyCLIP."""
    if category_table.empty:
        return category_table
    keys = ["backbone", "prompt_mode", "dataset", "corruption", "severity", "group"]
    aggregated = (category_table.groupby(keys, dropna=False)[list(ALL_METRICS)]
                  .mean().reset_index())
    aggregated["n_categories"] = (category_table.groupby(keys, dropna=False)["category"]
                                  .nunique().values)
    pooled = [pooled_image_metrics(config, row.backbone, row.prompt_mode, row.dataset,
                                   row.corruption, row.severity)
              for row in aggregated.itertuples()]
    return pd.concat([aggregated, pd.DataFrame(pooled)], axis=1)


def build_robustness_table(dataset_table: pd.DataFrame) -> pd.DataFrame:
    """Each corrupted cell beside its clean reference, plus the absolute drop."""
    if dataset_table.empty:
        return dataset_table
    clean = (dataset_table[dataset_table["corruption"] == "clean"]
             .set_index(["backbone", "prompt_mode", "dataset"])[list(ALL_METRICS)])
    corrupted = dataset_table[dataset_table["corruption"] != "clean"].copy()
    if clean.empty or corrupted.empty:
        return corrupted
    index = pd.MultiIndex.from_frame(
        corrupted[["backbone", "prompt_mode", "dataset"]])
    for metric in ALL_METRICS:
        reference = clean[metric].reindex(index).to_numpy()
        corrupted[f"{metric}_clean"] = reference
        corrupted[f"{metric}_drop"] = reference - corrupted[metric].to_numpy()
    return corrupted


def save_tables(config: BackboneEvalConfig,
                tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    os.makedirs(config.table_dir, exist_ok=True)
    written = {}
    for stem, table in tables.items():
        path = os.path.join(config.table_dir, f"{stem}_{config.fingerprint()}.csv")
        table.to_csv(path, index=False)
        written[stem] = path
        print(f"wrote {path}  ({len(table)} rows)")
    return written
