"""The sweep: fit prompts per source dataset, score every shard, write tables."""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from .aggregate import (build_dataset_table, build_robustness_table,
                        collect_category_table, save_tables)
from .artifacts import (archive_output, save_ground_truth, save_shard,
                        shard_is_done, write_run_manifest)
from .backbones import Backbone, backbone_errors, create_backbone
from .config import BackboneEvalConfig
from .corruptions import corruption_grid
from .datasets import categories_for, make_loader
from .determinism import seed_everything
from .metrics import evaluate_shard, resize_maps, resize_masks
from .prompts import FIXED_MODES, build_fixed_text
from .scoring import anomaly_outputs
from .training import train_prompts


def load_backbones(config: BackboneEvalConfig) -> dict[str, Backbone]:
    loaded: dict[str, Backbone] = {}
    for name in config.backbones:
        seed_everything(config.seed)
        try:
            backbone = create_backbone(name, config=config)
        except Exception as error:  # noqa: BLE001
            reason = backbone_errors().get(name, error)
            print(f"[skip] {name}: {reason}")
            continue
        loaded[name] = backbone
        print(f"{name:<10} {backbone.embed_dim:>5}-d | {backbone.depth} blocks | "
              f"dense layers {backbone.layers} | tau {backbone.temperature:.4f} | "
              f"{backbone.num_params / 1e6:.1f}M params | {backbone.image_size}px "
              f"| grid {backbone.grid}")
    return loaded


def sweep_plan(config: BackboneEvalConfig) -> list[dict]:
    plan = []
    for source, evaluate_on in config.protocol:
        for category in categories_for(config, evaluate_on):
            for corruption, severity in corruption_grid(config):
                plan.append({"source": source, "dataset": evaluate_on,
                             "category": category, "corruption": corruption,
                             "severity": severity})
    return plan


def run_shard(config: BackboneEvalConfig, backbone: Backbone, texts, dataset: str,
              category: str, corruption: str, severity: int,
              save: bool = True) -> tuple[dict, np.ndarray]:
    """Score one (backbone, dataset, category, corruption, severity) cell."""
    loader = make_loader(config, dataset, category, corruption, severity,
                         limit=config.limit)
    collected: dict[str, dict[str, list]] = {}
    masks, labels = [], []

    for batch in loader:
        outputs = anomaly_outputs(config, backbone, texts, batch["image"])
        for mode, (scores, maps) in outputs.items():
            store = collected.setdefault(mode, {"scores": [], "maps": []})
            store["scores"].append(scores.cpu().numpy().astype(np.float32))
            store["maps"].append(maps.cpu().numpy().astype(np.float16))
        masks.append(batch["mask"].numpy().astype(np.uint8))
        labels.append(batch["label"].numpy().astype(np.int64))

    masks = np.concatenate(masks)
    labels = np.concatenate(labels)
    results = {mode: {"scores": np.concatenate(store["scores"]),
                      "maps": np.concatenate(store["maps"]),
                      "labels": labels}
               for mode, store in collected.items()}

    # Scored here, at the full map resolution, because storage downsamples and
    # the detail cannot be recovered afterwards.
    for payload in results.values():
        payload["metrics"] = evaluate_shard(
            masks, payload["maps"], labels, payload["scores"],
            max_fpr=config.aupro_fpr_limit,
            num_thresholds=config.aupro_thresholds,
            ece_bins=config.ece_bins)
    if not save:
        return results, masks

    store_res = config.store_map_res or config.map_res
    save_ground_truth(config, dataset, category, corruption, severity,
                      resize_masks(masks, store_res), labels)
    for mode, payload in results.items():
        save_shard(config, backbone, mode, dataset, category, corruption, severity,
                   payload["scores"],
                   resize_maps(payload["maps"], store_res).astype(np.float16),
                   labels, payload["metrics"])
    return results, masks


def run_sweep(config: BackboneEvalConfig, backbones: dict[str, Backbone] | None = None,
              plan: list[dict] | None = None, verbose: bool = True) -> dict[str, Backbone]:
    backbones = load_backbones(config) if backbones is None else backbones
    plan = sweep_plan(config) if plan is None else plan
    total = len(plan) * max(len(backbones), 1)
    done = executed = 0
    started = time.time()

    for name, backbone in backbones.items():
        # One prompt set per source dataset; each is used only on the other one.
        learned_text = {
            source: train_prompts(config, backbone, source, verbose)().detach()
            for source, _ in config.protocol}

        frozen = [mode for mode in config.prompt_modes if mode in FIXED_MODES]
        # A class-agnostic ensemble never mentions the category, so it is the
        # same for all of them: build those once per backbone.
        static_text = {mode: build_fixed_text(config, backbone, "", mode=mode)
                       for mode in frozen if not FIXED_MODES[mode].category_dependent}
        per_category = [mode for mode in frozen
                        if FIXED_MODES[mode].category_dependent]

        current_key, category_text = None, {}
        for item in plan:
            done += 1
            if config.resume and config.limit is None and shard_is_done(
                    config, name, item["dataset"], item["category"],
                    item["corruption"], item["severity"]):
                continue

            key = (name, item["dataset"], item["category"])
            if key != current_key:
                category_text = {
                    mode: build_fixed_text(config, backbone, item["category"],
                                           mode=mode)
                    for mode in per_category}
                current_key = key

            texts = {**static_text, **category_text,
                     "learned": learned_text[item["source"]]}
            run_shard(config, backbone, texts, item["dataset"], item["category"],
                      item["corruption"], item["severity"],
                      save=(config.limit is None))
            executed += 1

            if verbose and executed % 20 == 0:
                elapsed = time.time() - started
                print(f"  {done}/{total} shards | {executed} computed | "
                      f"{elapsed / 60:.1f} min | {elapsed / executed:.1f}s per shard",
                      flush=True)

    if verbose:
        print(f"sweep complete: {executed} shards computed, "
              f"{total - executed} already present "
              f"({(time.time() - started) / 60:.1f} min)")
    return backbones


def run_evaluation(config: BackboneEvalConfig, verbose: bool = True) -> dict:
    """Load backbones, run the sweep, aggregate, and write every artefact."""
    seed_everything(config.seed)
    backbones = load_backbones(config)
    if not backbones:
        raise RuntimeError(
            f"no backbone could be loaded from {config.backbones}; "
            f"errors: {backbone_errors()}")

    manifest = write_run_manifest(
        config, {name: backbone.describe() for name, backbone in backbones.items()})
    if verbose:
        print(f"wrote {manifest}")

    run_sweep(config, backbones=backbones, verbose=verbose)

    category_table = collect_category_table(config, verbose)
    dataset_table = build_dataset_table(category_table)
    tables = {"category": category_table, "dataset": dataset_table,
              "robustness": build_robustness_table(dataset_table)}
    paths = save_tables(config, tables)

    result = {"config_id": config.fingerprint(), "manifest": manifest,
              "tables": paths, "n_category_rows": len(category_table)}
    if config.archive_results:
        result["archive"] = archive_output(config)
        if verbose:
            size_mb = os.path.getsize(result["archive"]) / 1e6
            print(f"wrote {result['archive']}  ({size_mb:.0f} MB)")
    return result
