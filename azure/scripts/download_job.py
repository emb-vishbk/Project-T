"""Download Azure ML job outputs to local artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _ws import get_ml_client


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Azure ML job outputs.")
    parser.add_argument("--job_name", required=True, help="Job name to download.")
    parser.add_argument(
        "--output_name",
        default="artifacts_root",
        help="Named output to download (ignored with --all).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all outputs.",
    )
    args = parser.parse_args()

    client, _ = get_ml_client()

    download_root = _project_root() / "artifacts" / "azure_downloads" / args.job_name
    download_root.mkdir(parents=True, exist_ok=True)

    if args.all:
        client.jobs.download(name=args.job_name, download_path=str(download_root), all=True)
    else:
        client.jobs.download(
            name=args.job_name,
            download_path=str(download_root),
            output_name=args.output_name,
        )

    print(f"Downloaded to: {download_root}")


if __name__ == "__main__":
    main()
