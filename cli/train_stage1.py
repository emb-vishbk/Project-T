"""Train Stage 1 encoder with a pluggable objective."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traceability.config.load import load_config
from traceability.stage1.train import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage 1 encoder.")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    parser.add_argument("--data_root", default=None, help="Override data root.")
    parser.add_argument("--artifacts_root", default=None, help="Override artifacts root.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        config["io"]["data_root"] = args.data_root
    if args.artifacts_root:
        config["io"]["artifacts_root"] = args.artifacts_root

    run_dir = train(config)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
