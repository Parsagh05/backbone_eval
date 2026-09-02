"""Kaggle entry point: edit the settings block, run, download the results ZIP.

Keeps the notebook thin -- the notebook's only job is to call this.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

# Both datasets are separate Kaggle datasets under different parents, so each
# root is given explicitly rather than discovered from one shared directory.
MVTEC_INPUT = "/kaggle/input/datasets/alirezasalehy/mvtec-ad/mvtec_anomaly_detection"
VISA_INPUT = "/kaggle/input/datasets/alirezasalehy/visa-ad/VisA_20220922"
OUTPUT_ROOT = "/kaggle/working/results"
WEIGHTS_DIR = "/kaggle/working/weights"

BACKBONES = ("clip", "siglip2")
SIGLIP2_DENSE_READOUT = "map_token"   # "raw" reproduces the published control
CORRUPTIONS_ENABLED = False           # clean-only for the backbone comparison
LIMIT = None                          # e.g. 8 for a smoke test
CATEGORIES = None                     # e.g. {"mvtec": ["hazelnut"], "visa": ["candle"]}
ARCHIVE = True


def build_config() -> dict:
    payload = {
        "mvtec_root": MVTEC_INPUT,
        "visa_root": VISA_INPUT,
        "output_root": OUTPUT_ROOT,
        "weights_dir": WEIGHTS_DIR,
        "backbones": list(BACKBONES),
        "siglip2_dense_readout": SIGLIP2_DENSE_READOUT,
        "corruptions_enabled": CORRUPTIONS_ENABLED,
        "limit": LIMIT,
        "device": "cuda",
    }
    if CATEGORIES:
        payload["categories"] = CATEGORIES
    return payload


def main() -> None:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src"))
    from bbeval import BackboneEvalConfig, run_evaluation

    config = BackboneEvalConfig(**build_config())
    result = run_evaluation(config)
    print(json.dumps(result, indent=2))

    if ARCHIVE:
        archive = shutil.make_archive(
            os.path.join("/kaggle/working", f"backbone_eval_{config.fingerprint()}"),
            "zip", OUTPUT_ROOT)
        print(f"wrote {archive}")


if __name__ == "__main__":
    main()
