"""Validated configuration for one backbone-benchmark invocation.

Every knob that was a notebook-level global lives here, so a run is fully
described by one JSON file and recorded alongside its artefacts.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

from ._version import __version__

# pptx slide 21: only the test splits are read, and the two roles are disjoint.
DEFAULT_PROTOCOL = (("mvtec", "visa"), ("visa", "mvtec"))

# pptx slides 18/19.
DEFAULT_CORRUPTION_GROUPS = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "blur": ("defocus_blur", "motion_blur", "zoom_blur"),
    "photometric": ("brightness", "contrast"),
    "geometric": ("rotation", "zoom_scale", "shift"),
}
DEFAULT_GEOMETRIC_MAGNITUDES = {
    "rotation": (5.0, 10.0, 15.0, 20.0),
    "zoom_scale": (1.05, 1.10, 1.15, 1.20),
    "shift": (0.02, 0.04, 0.06, 0.08),
}
DEFAULT_DENSE_LAYER_FRACTIONS = {
    "clip": (0.25, 0.5, 0.75, 1.0),
    # SigLIP2's standard output is the final encoder state followed by its MAP
    # attention-pooling head. Its paper does not prescribe AnomalyCLIP's four
    # intermediate stages for anomaly localisation.
    "siglip2": (1.0,),
}
# Frozen vocabularies, defined in prompts.py and mirrored here so the config can
# validate without importing it. prompts.py asserts the two agree.
FROZEN_PROMPT_MODES = ("fixed", "fixed_agnostic", "fixed_compact")
LEARNED_PROMPT_MODE = "learned"
VALID_PROMPT_MODES = FROZEN_PROMPT_MODES + (LEARNED_PROMPT_MODE,)


@dataclass
class BackboneEvalConfig:
    # --- paths ---------------------------------------------------------------
    mvtec_root: str
    visa_root: str
    output_root: str
    weights_dir: str | None = None

    # --- what to run ---------------------------------------------------------
    backbones: tuple[str, ...] = ("clip", "siglip2")
    protocol: tuple[tuple[str, str], ...] = DEFAULT_PROTOCOL
    categories: dict[str, tuple[str, ...]] | None = None
    prompt_modes: tuple[str, ...] = (
        "fixed", "fixed_agnostic", "fixed_compact", "learned")

    # --- backbone selection --------------------------------------------------
    clip_backbone: str = "ViT-L/14@336px"
    siglip2_model: str = "hf-hub:timm/ViT-L-16-SigLIP2-384"
    # "map_token" pools each patch through SigLIP2's own attention-pooling head,
    # the function that defines its joint image-text space. "raw" leaves trunk
    # tokens unprojected -- the CLIP-shaped assumption, kept as the control that
    # reproduces the published chance-level localisation.
    siglip2_dense_readout: str = "map_token"
    # None -> the checkpoint's native resolution (SigLIP2 is resolution-specific).
    siglip2_input_size: int | None = None

    # --- visual / scoring ----------------------------------------------------
    input_size: int = 518
    # Official AnomalyCLIP evaluates its maps at the configured 518px image
    # resolution. Lower values remain available as explicit storage/runtime
    # ablations, but are not the reference default.
    map_res: int = 518
    dense_layer_fractions: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: dict(DEFAULT_DENSE_LAYER_FRACTIONS))
    shared_dense_layers: tuple[float, ...] | None = None
    # Historical name retained for config compatibility. True now selects the
    # official AnomalyCLIP DPAM dense branch (V-V attention from layer 6), not
    # the former one-block value projection approximation.
    use_value_attention: bool = True
    global_token: str = "spatial"
    # AnomalyCLIP's image metric uses the global abnormal-text probability only.
    # Local fusion is retained solely as an explicit experimental ablation.
    add_local_evidence: bool = False
    gaussian_sigma: float = 4.0
    topk_fraction: float = 0.0

    # --- prompts -------------------------------------------------------------
    n_ctx: int = 12                          # AnomalyCLIP's default
    # AnomalyCLIP scales both image and patch similarities by 1/0.07 rather than
    # by the backbone's learned logit scale, and so does the group's own
    # object-agnostic-prompt-training. A shared value also gives both backbones
    # the same training-softmax sharpness. None uses each learned scale.
    map_temperature: float | None = 0.07
    # AnomalyCLIP initialises context vectors from N(0, 0.02), not from the
    # embeddings of the "X" placeholder tokens.
    init_std: float = 0.02
    learnable_suffix: dict[str, str] = field(
        default_factory=lambda: {"normal": "object.",
                                 "anomalous": "damaged object."})
    # None -> use the real category name, which is available zero-shot.
    fixed_prompt_class_name: str | None = None

    # --- prompt fitting ------------------------------------------------------
    # AnomalyCLIP's objective: image cross-entropy plus the mask-supervised
    # pixel terms. "local" drops the image term (Tipsomaly's localisation-only
    # ablation) and leaves nothing opposing a collapse onto "normal everywhere".
    loss_mode: str = "both"
    focal_gamma: float = 2.0
    focal_smooth: float = 1e-5
    image_loss_weight: float = 1.0
    pixel_loss_weight: float = 4.0          # AnomalyCLIP's lam
    # Keep CLIP's four-layer AnomalyCLIP inference map, but supervise only the
    # final selected layer during shallow-prompt fitting. SigLIP2 already reads
    # only its final layer, so both backbones receive one local loss term.
    pixel_loss_layers: str = "last"
    # AnomalyCLIP's setting. Two epochs was not enough to leave the
    # "normal everywhere" basin the pixel loss starts in.
    epochs: int = 15
    batch_size: int = 8
    learning_rate: float = 1e-3
    adam_betas: tuple[float, float] = (0.5, 0.999)
    weight_decay: float = 0.0
    # None matches the reference shallow-prompt trainer. A positive value is
    # retained as an explicit ablation knob rather than silently changing the
    # default optimisation path.
    grad_clip: float | None = None
    max_train_images_per_category: int | None = None

    # --- corruptions (pptx slides 18/19/21) ----------------------------------
    corruption_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_CORRUPTION_GROUPS))
    geometric_magnitudes: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: dict(DEFAULT_GEOMETRIC_MAGNITUDES))
    severities: tuple[int, ...] = (1, 2, 3)
    include_clean: bool = True
    corruptions_enabled: bool = True

    # --- metrics -------------------------------------------------------------
    aupro_fpr_limit: float = 0.30
    aupro_thresholds: int = 200
    ece_bins: int = 15

    # --- runtime -------------------------------------------------------------
    device: str = "cuda"
    seed: int = 111
    num_workers: int = 0
    amp: bool = True
    resume: bool = True
    limit: int | None = None
    # Collect every artefact into one ZIP beside output_root, so a Kaggle run
    # leaves a single file to download.
    archive_results: bool = True
    run_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.backbones = tuple(self.backbones)
        self.protocol = tuple(tuple(pair) for pair in self.protocol)
        self.prompt_modes = tuple(self.prompt_modes)
        self.severities = tuple(int(value) for value in self.severities)
        self.adam_betas = tuple(float(value) for value in self.adam_betas)
        self.corruption_groups = {
            key: tuple(value) for key, value in self.corruption_groups.items()}
        self.geometric_magnitudes = {
            key: tuple(value) for key, value in self.geometric_magnitudes.items()}
        self.dense_layer_fractions = {
            key: tuple(value) for key, value in self.dense_layer_fractions.items()}
        if self.shared_dense_layers is not None:
            self.shared_dense_layers = tuple(self.shared_dense_layers)
        if self.categories is not None:
            self.categories = {
                key: tuple(value) for key, value in self.categories.items()}

        if self.loss_mode not in ("local", "global", "both"):
            raise ValueError(
                f"loss_mode must be local, global or both; got {self.loss_mode!r}")
        if self.pixel_loss_layers not in ("last", "all"):
            raise ValueError(
                "pixel_loss_layers must be last or all; got "
                f"{self.pixel_loss_layers!r}")
        if self.siglip2_dense_readout not in ("map_token", "raw"):
            raise ValueError(
                "siglip2_dense_readout must be map_token or raw; got "
                f"{self.siglip2_dense_readout!r}")
        unknown = set(self.prompt_modes) - set(VALID_PROMPT_MODES)
        if unknown:
            raise ValueError(f"unknown prompt mode(s): {sorted(unknown)}")
        if not self.prompt_modes:
            raise ValueError("prompt_modes cannot be empty")
        # pptx slide 23 asks for both frozen and learnable prompt performance,
        # but a single run may legitimately cover one half: shards are stored
        # per mode, so a later run into the same output_root completes the pair,
        # and a frozen-only run skips prompt fitting entirely. Warn, do not fail.
        missing = []
        if not set(self.prompt_modes) & set(FROZEN_PROMPT_MODES):
            missing.append(f"a frozen mode from {FROZEN_PROMPT_MODES}")
        if LEARNED_PROMPT_MODE not in self.prompt_modes:
            missing.append(repr(LEARNED_PROMPT_MODE))
        if missing:
            warnings.warn(
                f"prompt_modes={self.prompt_modes} omits {' and '.join(missing)}; "
                "slide 23 asks for both frozen and learnable results, so this "
                "run covers only part of the protocol.",
                stacklevel=2)
        if self.input_size <= 0 or self.map_res <= 0:
            raise ValueError("input_size and map_res must be positive")
        if self.n_ctx <= 0:
            raise ValueError("n_ctx must be positive")
        if self.map_temperature is not None and self.map_temperature <= 0:
            raise ValueError("map_temperature must be positive or None")
        if self.init_std <= 0:
            raise ValueError("init_std must be positive")
        if not 0 <= self.focal_smooth < 0.5:
            raise ValueError("focal_smooth must be in [0, 0.5)")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive or None")

    # --- derived -------------------------------------------------------------
    @property
    def dataset_roots(self) -> dict[str, str]:
        return {"mvtec": self.mvtec_root, "visa": self.visa_root}

    @property
    def artifact_dir(self) -> str:
        return f"{self.output_root}/artifacts"

    @property
    def checkpoint_dir(self) -> str:
        return f"{self.output_root}/prompts"

    @property
    def table_dir(self) -> str:
        return f"{self.output_root}/tables"

    def fingerprint(self) -> str:
        """Identifies the settings a stored artefact was produced under."""
        payload = {
            # A behaviour change must invalidate earlier artefacts rather than
            # resume on top of them, so the code revision is part of the id.
            "revision": __version__,
            "seed": self.seed,
            "input": self.input_size,
            "map_res": self.map_res,
            "backbones": list(self.backbones),
            "clip": self.clip_backbone,
            "siglip2": self.siglip2_model,
            "siglip2_readout": self.siglip2_dense_readout,
            "siglip2_input": self.siglip2_input_size,
            "layers": (self.dense_layer_fractions if self.shared_dense_layers is None
                       else {"shared": self.shared_dense_layers}),
            "value_attention": self.use_value_attention,
            "n_ctx": self.n_ctx,
            "learnable_suffix": self.learnable_suffix,
            "fixed_prompt_class_name": self.fixed_prompt_class_name,
            "loss": self.loss_mode,
            "global_token": self.global_token,
            "local_evidence": self.add_local_evidence,
            "sigma": self.gaussian_sigma,
            # Prompt fitting changes the learned map, so it changes the id.
            "epochs": self.epochs,
            "lr": self.learning_rate,
            "train_cap": self.max_train_images_per_category,
            "image_weight": self.image_loss_weight,
            "pixel_weight": self.pixel_loss_weight,
            "pixel_loss_layers": self.pixel_loss_layers,
            "focal_gamma": self.focal_gamma,
            "focal_smooth": self.focal_smooth,
            "map_temperature": self.map_temperature,
            "init_std": self.init_std,
            "batch_size": self.batch_size,
            "adam_betas": self.adam_betas,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.blake2b(blob.encode(), digest_size=6).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
