"""JSON-configured command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backbones import backbone_errors, backbone_names
from .config import BackboneEvalConfig
from .engine import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backbone benchmarking for zero-shot anomaly detection")
    parser.add_argument("--config", help="Path to a BackboneEvalConfig JSON file")
    parser.add_argument("--list-backbones", action="store_true",
                        help="List registered backbones and why any are unavailable")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                        help="Override one config field, e.g. --set limit=4")
    args = parser.parse_args()

    if args.list_backbones:
        for name in backbone_names():
            print(name)
        for name, reason in backbone_errors().items():
            print(f"{name}\t[unavailable] {reason}")
        return

    if not args.config:
        parser.error("--config is required unless --list-backbones is used")

    payload = json.loads(Path(args.config).expanduser().resolve()
                         .read_text(encoding="utf-8"))
    for override in args.set:
        key, _, raw = override.partition("=")
        if not _:
            parser.error(f"--set expects KEY=JSON, got {override!r}")
        try:
            payload[key] = json.loads(raw)
        except json.JSONDecodeError:
            payload[key] = raw
    print(json.dumps(run_evaluation(BackboneEvalConfig(**payload)), indent=2))


if __name__ == "__main__":
    main()
