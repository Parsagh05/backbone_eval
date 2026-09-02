"""End-to-end run of the whole pipeline on synthetic data.

Uses a stub backbone so the sweep, prompt fitting, artefact round-trip, metrics
and aggregation are all exercised without weights, a GPU, or the real datasets.
This is the gate that the modules actually compose -- the per-backbone forward
paths are covered separately in test_siglip2_backbone.py.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bbeval.aggregate import build_dataset_table, collect_category_table  # noqa: E402
from bbeval.artifacts import load_shard  # noqa: E402
from bbeval.backbones.base import Backbone, register_backbone  # noqa: E402
from bbeval.config import BackboneEvalConfig  # noqa: E402
from bbeval.engine import run_evaluation, sweep_plan  # noqa: E402
from bbeval.metrics import ALL_METRICS  # noqa: E402

IMAGE_SIZE = 32
PATCH = 8
EMBED = 16
N_NORMAL, N_DEFECT = 3, 3


# --- a backbone that needs no weights ----------------------------------------
class StubBackbone(Backbone):
    """Deterministic stand-in implementing the full frozen-encoder contract."""

    name = "stub"
    has_two_global_tokens = False

    def __init__(self, config: BackboneEvalConfig) -> None:
        self.config = config
        self.device = config.device
        self.embed_dim = EMBED
        self.depth = 2
        self.layers = (2,)
        self.temperature = 0.07
        self.image_size = IMAGE_SIZE
        self.patch_size = PATCH
        self.grid = IMAGE_SIZE // PATCH
        self.num_params = 0
        torch.manual_seed(0)
        # A frozen module, so assert_prompt_learning_only has something to check.
        self.model = torch.nn.Linear(EMBED, EMBED)
        self.model.requires_grad_(False)
        self._project = torch.nn.Parameter(torch.randn(EMBED, EMBED) * 0.1,
                                           requires_grad=False)

    def preprocess(self, images_uint8):
        return images_uint8.float().div(255.0)

    def encode(self, x):
        # Brightness per patch -> a feature that actually varies with the image,
        # so metrics are not degenerate.
        batch = x.shape[0]
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, self.grid)
        tokens = pooled.permute(0, 2, 3, 1).reshape(batch, self.grid * self.grid, 3)
        dense = torch.nn.functional.pad(tokens, (0, EMBED - 3))
        dense = dense.reshape(batch, self.grid, self.grid, EMBED)
        return {"object": dense.mean(dim=(1, 2)),
                "spatial": dense.mean(dim=(1, 2)),
                "dense": {2: dense}}

    def init_prompt(self, suffixes):
        ctx = torch.randn(len(suffixes), self.config.n_ctx, EMBED) * 0.02
        return ctx, {"tail": None}

    def encode_text(self, ctx, aux):
        return ctx.mean(dim=1) @ self._project        # differentiable in ctx

    def encode_fixed_text(self, texts):
        generator = torch.Generator().manual_seed(len(texts))
        return torch.randn(len(texts), EMBED, generator=generator)


register_backbone("stub")(StubBackbone)


# --- synthetic datasets ------------------------------------------------------
def _write_png(path, value):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    array = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def _write_mask(path):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    array = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    array[8:24, 8:24] = 255                    # one connected region, for AUPRO
    Image.fromarray(array).save(path)


def build_mvtec(root, category="widget"):
    for index in range(N_NORMAL):
        _write_png(os.path.join(root, category, "test", "good", f"{index:03d}.png"), 40)
    for index in range(N_DEFECT):
        _write_png(os.path.join(root, category, "test", "broken", f"{index:03d}.png"), 200)
        _write_mask(os.path.join(root, category, "ground_truth", "broken",
                                 f"{index:03d}_mask.png"))


def build_visa(root, category="gizmo"):
    rows = []
    for index in range(N_NORMAL):
        rel = f"{category}/Data/Images/Normal/{index:04d}.JPG"
        _write_png(os.path.join(root, rel), 40)
        rows.append({"object": category, "split": "test", "label": "normal",
                     "image": rel, "mask": ""})
    for index in range(N_DEFECT):
        rel = f"{category}/Data/Images/Anomaly/{index:04d}.JPG"
        mask_rel = f"{category}/Data/Masks/Anomaly/{index:04d}.png"
        _write_png(os.path.join(root, rel), 200)
        _write_mask(os.path.join(root, mask_rel))
        rows.append({"object": category, "split": "test", "label": "bad",
                     "image": rel, "mask": mask_rel})
    split_dir = os.path.join(root, "split_csv")
    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, "1cls.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["object", "split", "label",
                                                    "image", "mask"])
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(scope="module")
def config(tmp_path_factory) -> BackboneEvalConfig:
    root = tmp_path_factory.mktemp("pipeline")
    mvtec, visa = str(root / "mvtec"), str(root / "visa")
    build_mvtec(mvtec)
    build_visa(visa)
    return BackboneEvalConfig(
        mvtec_root=mvtec, visa_root=visa, output_root=str(root / "out"),
        backbones=("stub",), categories={"mvtec": ("widget",), "visa": ("gizmo",)},
        input_size=IMAGE_SIZE, map_res=8, n_ctx=4, epochs=1, batch_size=2,
        corruptions_enabled=False, device="cpu", amp=False, num_workers=0,
        dense_layer_fractions={"stub": (1.0,)}, aupro_thresholds=16)


@pytest.fixture(scope="module")
def result(config):
    return run_evaluation(config, verbose=False)


def test_sweep_plan_is_the_disjoint_cross_protocol(config):
    plan = sweep_plan(config)
    # Two protocol directions, one category each, clean only.
    assert len(plan) == 2
    assert {(item["source"], item["dataset"]) for item in plan} == {
        ("mvtec", "visa"), ("visa", "mvtec")}
    # Prompts are never fitted and evaluated on the same dataset.
    assert all(item["source"] != item["dataset"] for item in plan)


def test_run_writes_manifest_and_tables(result, config):
    assert os.path.isfile(result["manifest"])
    for stem in ("category", "dataset"):
        assert os.path.isfile(result["tables"][stem])
    assert result["config_id"] == config.fingerprint()


def test_archive_bundles_every_artefact(result, config):
    """One ZIP to download, written beside output_root rather than inside it."""
    import zipfile

    archive = result["archive"]
    assert os.path.isfile(archive)
    # Inside output_root the archive would try to include itself while writing.
    assert not os.path.abspath(archive).startswith(
        os.path.abspath(config.output_root) + os.sep)

    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
    assert any(name.startswith("tables/") and name.endswith(".csv") for name in names)
    assert any(name.startswith("prompts/") and name.endswith(".pt") for name in names)
    assert any(name.startswith("artifacts/stub/fixed/") for name in names)
    assert any(name.startswith("artifacts/_ground_truth/") for name in names)
    assert any(name.startswith("run_manifest_") for name in names)


def test_shards_round_trip_for_every_prompt_mode(result, config):
    for mode in config.prompt_modes:
        shard = load_shard(config, "stub", mode, "mvtec", "widget", "clean", 0)
        assert shard is not None, mode
        assert shard["scores"].shape == (N_NORMAL + N_DEFECT,)
        assert shard["maps"].shape == (N_NORMAL + N_DEFECT, config.map_res, config.map_res)
        assert set(np.unique(shard["labels"])) == {0, 1}
        assert shard["meta"]["backbone"] == "stub"
        assert shard["meta"]["config_id"] == config.fingerprint()


def test_metrics_are_finite_and_in_range(result, config):
    table = collect_category_table(config, verbose=False)
    assert not table.empty
    assert set(table["dataset"]) == {"mvtec", "visa"}
    for metric in ALL_METRICS:
        values = table[metric].to_numpy(dtype=float)
        assert np.isfinite(values).all(), metric
        if metric.endswith(("auroc", "f1max", "aupro", "ap")):
            assert ((values >= 0.0) & (values <= 1.0)).all(), metric


def test_dataset_table_aggregates_over_categories(config):
    category_table = collect_category_table(config, verbose=False)
    dataset_table = build_dataset_table(config, category_table)
    assert not dataset_table.empty
    assert (dataset_table["n_categories"] == 1).all()
    assert "image_auroc_pooled" in dataset_table.columns


def test_prompt_checkpoint_contains_only_context(config):
    from bbeval.training import checkpoint_path
    path = checkpoint_path(config, "stub", "mvtec")
    assert os.path.isfile(path)
    state = torch.load(path, map_location="cpu")
    assert list(state) == ["ctx"]
    assert state["ctx"].shape == (2, config.n_ctx, EMBED)


def test_resume_skips_completed_shards(config):
    from bbeval.engine import load_backbones, run_sweep

    def mtimes():
        return {(item["dataset"], item["category"]): os.path.getmtime(
            os.path.join(config.artifact_dir, "stub", "fixed", item["dataset"],
                         item["category"], "clean_s0.npz"))
            for item in sweep_plan(config)}

    # Everything is already on disk, so a second sweep must rewrite nothing.
    before = mtimes()
    run_sweep(config, backbones=load_backbones(config), verbose=False)
    assert mtimes() == before
