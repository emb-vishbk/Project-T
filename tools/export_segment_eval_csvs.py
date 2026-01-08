"""Export per-session segment summaries from Stage-2 assignments.

Segments are runs of identical cluster_id_smooth. Segment times are derived
from t_end using round(t_end / fs_hz) to align with video timestamps.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple


REQUIRED_COLUMNS = {"t_end", "cluster_id_smooth"}


def _seconds_to_mmss(seconds: float) -> str:
    total_sec = int(round(seconds))
    if total_sec < 0:
        total_sec = 0
    mm = total_sec // 60
    ss = total_sec % 60
    return f"{mm:02d}:{ss:02d}"


def _load_t_end_and_cluster(path: Path) -> Tuple[List[int], List[int]]:
    if path.stat().st_size == 0:
        return [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return [], []
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing columns in {path}: {missing_list}")
        t_end: List[int] = []
        clusters: List[int] = []
        for row in reader:
            if not row:
                continue
            try:
                t_end.append(int(float(row["t_end"])))
                clusters.append(int(float(row["cluster_id_smooth"])))
            except ValueError as exc:
                raise ValueError(f"Invalid values in {path}") from exc
    return t_end, clusters


def _segments_from_clusters(
    t_end: List[int],
    clusters: List[int],
    fs_hz: float,
) -> List[Tuple[int, str, str, int]]:
    if not clusters:
        return []
    if len(t_end) != len(clusters):
        raise ValueError("t_end and cluster_id_smooth lengths do not match")

    segments: List[Tuple[int, str, str, int]] = []
    start = 0
    seg_no = 0
    for i in range(1, len(clusters)):
        if clusters[i] != clusters[i - 1]:
            start_time = _seconds_to_mmss(t_end[start] / fs_hz)
            end_time = _seconds_to_mmss(t_end[i - 1] / fs_hz)
            segments.append((seg_no, start_time, end_time, clusters[start]))
            seg_no += 1
            start = i
    start_time = _seconds_to_mmss(t_end[start] / fs_hz)
    end_time = _seconds_to_mmss(t_end[-1] / fs_hz)
    segments.append((seg_no, start_time, end_time, clusters[start]))
    return segments


def _write_segments(path: Path, segments: Iterable[Tuple[int, str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment_no", "segment_start", "segment_end", "cluster_id_smooth"])
        for row in segments:
            writer.writerow(row)


def _assignment_dirs(stage2_dir: Path) -> List[Path]:
    dirs: List[Path] = []
    for child in stage2_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("assignments_") and not name.endswith("_eval"):
            dirs.append(child)
    return sorted(dirs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export segment summary CSVs from per-window assignments."
    )
    parser.add_argument(
        "--stage2_dir",
        type=str,
        required=True,
        help="Stage-2 output root (e.g., artifacts/stage2/kmeans_k16_pca).",
    )
    parser.add_argument(
        "--fs_hz",
        type=float,
        default=3.0,
        help="Sampling rate in Hz used to convert t_end to time (default: 3.0).",
    )
    args = parser.parse_args()

    stage2_dir = Path(args.stage2_dir)
    if not stage2_dir.exists():
        raise FileNotFoundError(f"Stage-2 dir not found: {stage2_dir}")

    eval_root = stage2_dir / "evaluations"
    for assignments_dir in _assignment_dirs(stage2_dir):
        split = assignments_dir.name.replace("assignments_", "", 1)
        out_dir = eval_root / f"assignments_{split}_eval"
        for csv_path in sorted(assignments_dir.glob("*.csv")):
            t_end, clusters = _load_t_end_and_cluster(csv_path)
            segments = _segments_from_clusters(t_end, clusters, args.fs_hz)
            out_path = out_dir / csv_path.name
            _write_segments(out_path, segments)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
