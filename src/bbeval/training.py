"""Fitting the prompt context on a source dataset's test split."""

from __future__ import annotations

import os
import time

import torch

from .backbones import Backbone
from .config import BackboneEvalConfig
from .datasets import make_train_loader
from .determinism import derived_seed, seed_everything
from .losses import prompt_loss
from .prompts import LearnablePrompts
from .scoring import training_logits

# Below this peak abnormal probability the map is constant for every practical
# purpose, and every pixel metric degenerates to chance.
COLLAPSE_PEAK = 0.01


def checkpoint_path(config: BackboneEvalConfig, backbone_name: str, source: str) -> str:
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    return os.path.join(
        config.checkpoint_dir,
        f"{backbone_name}_{source}_{config.loss_mode}_ctx{config.n_ctx}"
        f"_ep{config.epochs}_seed{config.seed}.pt")


def assert_prompt_learning_only(config: BackboneEvalConfig, backbone: Backbone,
                                prompts: LearnablePrompts) -> list[torch.nn.Parameter]:
    """Protocol constraint: encoders frozen; only prompt context vectors train."""
    for attribute in ("model", "visual", "image_model", "text_model"):
        module = getattr(backbone, attribute, None)
        if not isinstance(module, torch.nn.Module):
            continue
        for parameter in module.parameters():
            if parameter.requires_grad:
                raise AssertionError(
                    f"protocol violated: {attribute} has trainable encoder "
                    "parameters (prompt learning only)")
    trainable = [p for p in prompts.parameters() if p.requires_grad]
    expected = 2 * config.n_ctx * backbone.embed_dim
    if len(trainable) != 1 or trainable[0].numel() != expected:
        raise AssertionError(
            f"expected exactly one trainable tensor of {expected} elements, got "
            f"{[p.numel() for p in trainable]}")
    return trainable


def train_prompts(config: BackboneEvalConfig, backbone: Backbone, source: str,
                  verbose: bool = True) -> LearnablePrompts:
    """Fit the context vectors on `source`. Resumes from a checkpoint if present."""
    path = checkpoint_path(config, backbone.name, source)
    prompts = LearnablePrompts(config, backbone).to(config.device)
    trainable = assert_prompt_learning_only(config, backbone, prompts)

    if config.resume and os.path.isfile(path):
        prompts.load_state_dict(torch.load(path, map_location=config.device))
        if verbose:
            print(f"  [{backbone.name}/{source}] loaded {os.path.basename(path)}")
        return prompts.eval()

    seed_everything(derived_seed(config.seed, "train", backbone.name, source))
    loader = make_train_loader(config, source)
    optimizer = torch.optim.Adam(trainable, lr=config.learning_rate,
                                 betas=config.adam_betas,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs * len(loader)))

    if verbose:
        print(f"  [{backbone.name}/{source}] fitting {trainable[0].numel():,} prompt "
              f"parameters on {len(loader.dataset)} images "
              f"({config.epochs} epochs, {config.loss_mode} loss)")
    prompts.train()
    peak = 0.0
    for epoch in range(config.epochs):
        started, running, seen, average, peak = time.time(), 0.0, 0, 0.0, 0.0
        for batch in loader:
            image_logits, pixel_logits = training_logits(
                config, backbone, prompts, batch["image"])
            loss = prompt_loss(config, image_logits, pixel_logits,
                               batch["label"].to(config.device),
                               batch["mask"].to(config.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            optimizer.step()
            scheduler.step()
            count = batch["label"].shape[0]
            running += loss.item() * count
            seen += count
            # The abnormal channel is what becomes the anomaly map. Watching it
            # here is the difference between noticing a collapse now and
            # discovering a table of 50.0 pixel AUROC hours later.
            with torch.no_grad():
                # Averaged over layers, which is what inference scores.
                merged = pixel_logits.mean(0) if pixel_logits.ndim == 5 else pixel_logits
                abnormal = merged.softmax(dim=1)[:, 1]
                average += float(abnormal.mean()) * count
                peak = max(peak, float(abnormal.max()))
        if verbose:
            print(f"    epoch {epoch + 1}/{config.epochs}  "
                  f"loss {running / max(seen, 1):.4f}  "
                  f"map mean {average / max(seen, 1):.4f} peak {peak:.4f}  "
                  f"({time.time() - started:.0f}s)")

    if peak < COLLAPSE_PEAK:
        print(f"  WARNING [{backbone.name}/{source}] the abnormal channel never "
              f"exceeded {peak:.2e}: the prompts have collapsed onto 'normal "
              f"everywhere' and the anomaly map will be constant. Pixel AUROC "
              f"will come out at chance. Try more epochs or a lower "
              f"learning rate.", flush=True)

    prompts.eval()
    torch.save(prompts.state_dict(), path)
    return prompts
