"""Extract embeddings for Stage 1 encoder runs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traceability.stage1.extract_embeddings import extract_embeddings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract embeddings from a trained encoder.")
    parser.add_argument("--run_dir", required=True, help="Run directory containing config/weights.")
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated list of splits.",
    )
    parser.add_argument("--data_root", default=None, help="Override data root.")
    parser.add_argument("--artifacts_root", default=None, help="Override artifacts root.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    extract_embeddings(
        run_dir=args.run_dir,
        splits=splits,
        data_root_override=args.data_root,
        artifacts_root_override=args.artifacts_root,
    )
    print("Embedding extraction complete.")


if __name__ == "__main__":
    main()
