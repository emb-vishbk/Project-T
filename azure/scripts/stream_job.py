"""Stream logs for an Azure ML job or print status."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _ws import get_ml_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Azure ML job logs.")
    parser.add_argument("--job_name", required=True, help="Job name to stream.")
    parser.add_argument(
        "--status_only",
        action="store_true",
        help="Print job status and exit.",
    )
    args = parser.parse_args()

    client, _ = get_ml_client()
    if args.status_only:
        job = client.jobs.get(args.job_name)
        print(f"{args.job_name}: {job.status}")
        return

    client.jobs.stream(args.job_name)


if __name__ == "__main__":
    main()
