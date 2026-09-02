"""Seeding that does not depend on execution order."""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def derived_seed(seed: int, *parts: object) -> int:
    """A stable seed for a named piece of work, independent of run order."""
    key = "|".join(str(part) for part in (seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def dataloader_kwargs(seed: int) -> dict:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def worker_init_fn(worker_id: int) -> None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)

    return {"generator": generator, "worker_init_fn": worker_init_fn}
