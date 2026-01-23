"""List Azure ML environments to choose a curated environment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _ws import get_ml_client


def _parse_filters(raw: list[str]) -> list[str]:
    filters: list[str] = []
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if part:
                filters.append(part.lower())
    return filters


def _matches(text: str, filters: list[str]) -> bool:
    text = text.lower()
    return all(filt in text for filt in filters)


def main() -> None:
    parser = argparse.ArgumentParser(description="List Azure ML environments.")
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        help="Substring filter (can be repeated or comma-separated).",
    )
    args = parser.parse_args()

    filters = _parse_filters(args.contains)
    client, _ = get_ml_client()

    envs = []
    for env in client.environments.list():
        name = getattr(env, "name", "")
        version = getattr(env, "version", "")
        description = getattr(env, "description", "") or ""
        tags = " ".join(f"{k}:{v}" for k, v in (getattr(env, "tags", {}) or {}).items())
        haystack = " ".join([name, description, tags])
        if filters and not _matches(haystack, filters):
            continue
        envs.append((name, version))

    envs.sort(key=lambda item: (item[0].lower(), str(item[1])))

    if not envs:
        print("No environments matched.")
        return

    for name, version in envs:
        print(f"{name}@{version}")


if __name__ == "__main__":
    main()
