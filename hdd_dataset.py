"""Utilities for preparing and loading the HDD 3 Hz subset.

This module stays close to the constraints in AGENTS.md:
- sliding windows are defined by window length (w) and hop
- indices store only (session_id, t_end) to avoid duplicating data
- splits happen at the session level
- labels are evaluation-only
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

FS_HZ = 3.0
DEFAULT_WINDOW = 18  # 6 seconds at 3 Hz
DEFAULT_HOP_TRAIN = 3  # 1 second
DEFAULT_HOP_INFER = 1  # 0.33 second
CHANNEL_NAMES = [
    "accel_pedal_pct",
    "steer_angle_deg",
    "steer_speed",
    "speed",
    "brake_kpa",
    "lturn",
    "rturn",
    "yaw_deg_s",
]
BINARY_CHANNEL_NAMES = {"lturn", "rturn"}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_npy(path: Path, mmap: bool = True) -> np.ndarray:
    mode = "r" if mmap else None
    return np.load(str(path), mmap_mode=mode)


def _save_json(obj: dict, path: Path) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))


def _save_jsonl(records: Iterable[dict], path: Path) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def list_session_ids(directory: Union[str, Path]) -> List[str]:
    """Return sorted session_ids (stem of .npy files) from a directory."""
    directory = Path(directory)
    return sorted(p.stem for p in directory.glob("*.npy"))


def scan_dataset(
    sensor_dir: Union[str, Path],
    label_dir: Union[str, Path],
    window: int = DEFAULT_WINDOW,
) -> dict:
    """Scan dataset files and return a summary dictionary."""
    sensor_dir = Path(sensor_dir)
    label_dir = Path(label_dir)
    sensor_ids = set(list_session_ids(sensor_dir))
    label_ids = set(list_session_ids(label_dir))
    common_ids = sorted(sensor_ids & label_ids)
    missing_sensors = sorted(label_ids - sensor_ids)
    missing_labels = sorted(sensor_ids - label_ids)

    lengths: List[int] = []
    durations_min: List[float] = []
    dtype_set: set = set()
    label_counts: Counter = Counter()
    per_session_label_counts: Dict[str, Dict[str, int]] = {}
    mismatched_lengths: Dict[str, Tuple[int, int]] = {}
    nan_sessions: List[str] = []

    for session_id in common_ids:
        x = _load_npy(sensor_dir / f"{session_id}.npy")
        y = _load_npy(label_dir / f"{session_id}.npy")
        dtype_set.add(str(x.dtype))

        if x.shape[0] != y.shape[0]:
            mismatched_lengths[session_id] = (int(x.shape[0]), int(y.shape[0]))
            continue

        if np.isnan(x).any() or np.isnan(y).any():
            nan_sessions.append(session_id)

        n = int(x.shape[0])
        lengths.append(n)
        durations_min.append(n / FS_HZ / 60.0)

        unique, counts = np.unique(y, return_counts=True)
        per_session_label_counts[session_id] = {
            str(int(u)): int(c) for u, c in zip(unique, counts)
        }
        label_counts.update(dict(zip(unique.astype(int), counts)))

    total_minutes = sum(durations_min)
    summary = {
        "num_sessions": len(common_ids),
        "sensor_only_sessions": missing_labels,
        "label_only_sessions": missing_sensors,
        "lengths": {
            "min": int(min(lengths)) if lengths else 0,
            "max": int(max(lengths)) if lengths else 0,
            "mean": float(np.mean(lengths)) if lengths else 0.0,
            "median": float(np.median(lengths)) if lengths else 0.0,
        },
        "durations_minutes": {
            "total": total_minutes,
            "min": float(min(durations_min)) if durations_min else 0.0,
            "max": float(max(durations_min)) if durations_min else 0.0,
            "mean": float(np.mean(durations_min)) if durations_min else 0.0,
            "median": float(np.median(durations_min)) if durations_min else 0.0,
        },
        "feature_dtypes": sorted(dtype_set),
        "label_inventory": [int(k) for k in sorted(label_counts.keys())],
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "per_session_label_counts": per_session_label_counts,
        "mismatched_lengths": mismatched_lengths,
        "nan_sessions": nan_sessions,
        "window": window,
        "fs_hz": FS_HZ,
    }

    print(f"Sessions with sensors+labels: {len(common_ids)}")
    if missing_sensors:
        print(f"Missing sensors for {len(missing_sensors)} sessions (labels only).")
    if missing_labels:
        print(f"Missing labels for {len(missing_labels)} sessions (sensors only).")
    if mismatched_lengths:
        print(f"Mismatched lengths in {len(mismatched_lengths)} sessions.")
    print(f"Lengths (frames): min {summary['lengths']['min']}, "
          f"max {summary['lengths']['max']}, mean {summary['lengths']['mean']:.2f}")
    print(f"Total duration: {total_minutes:.2f} minutes")
    print(f"Label inventory: {summary['label_inventory']}")

    return summary


def build_session_splits(
    session_ids: Sequence[str],
    seed: int = 123,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Dict[str, List[str]]:
    """Split session_ids into train/val/test lists with a fixed seed."""
    session_ids = list(session_ids)
    rng = random.Random(seed)
    rng.shuffle(session_ids)

    n = len(session_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_ids = session_ids[:n_train]
    val_ids = session_ids[n_train : n_train + n_val]
    test_ids = session_ids[n_train + n_val :]

    return {"train": train_ids, "val": val_ids, "test": test_ids, "seed": seed}


def _window_indices_for_length(length: int, window: int, hop: int) -> List[int]:
    if length < window:
        return []
    start = window - 1
    return list(range(start, length, hop))


def build_window_index(
    session_ids: Sequence[str],
    sensor_dir: Union[str, Path],
    window: int,
    hop: int,
) -> List[Dict[str, int]]:
    """Return list of {"session_id": str, "t_end": int}."""
    sensor_dir = Path(sensor_dir)
    index: List[Dict[str, int]] = []
    for session_id in session_ids:
        x = _load_npy(sensor_dir / f"{session_id}.npy")
        for t_end in _window_indices_for_length(int(x.shape[0]), window, hop):
            index.append({"session_id": session_id, "t_end": int(t_end)})
    return index


def compute_normalization(
    session_ids: Sequence[str],
    sensor_dir: Union[str, Path],
    channel_names: Sequence[str] = CHANNEL_NAMES,
    eps: float = 1e-8,
) -> dict:
    """Compute per-channel mean/std using train sessions only.

    Binary indicator channels (lturn/rturn) should remain 0/1, so they are not standardized.
    We encode that by writing mean=0 and std=1 for those channels in the returned schema.
    """
    sensor_dir = Path(sensor_dir)
    channel_names = list(channel_names)
    n_channels = len(channel_names)
    sum_ = np.zeros(n_channels, dtype=np.float64)
    sumsq = np.zeros(n_channels, dtype=np.float64)
    count = np.zeros(n_channels, dtype=np.int64)
    nan_sessions: List[str] = []

    binary_indices = [i for i, name in enumerate(channel_names) if name in BINARY_CHANNEL_NAMES]

    for session_id in session_ids:
        x = _load_npy(sensor_dir / f"{session_id}.npy")
        nan_mask = np.isnan(x)
        if nan_mask.any():
            nan_sessions.append(session_id)
        valid_mask = ~nan_mask
        valid_counts = valid_mask.sum(axis=0)
        # avoid reallocation by zeroing NaNs before summation
        x_clean = np.where(valid_mask, x, 0.0)
        sum_ += x_clean.sum(axis=0)
        sumsq += (x_clean * x_clean).sum(axis=0)
        count += valid_counts

    safe_count = np.where(count == 0, 1, count)
    mean = sum_ / safe_count
    var = sumsq / safe_count - mean * mean
    var = np.clip(var, 0.0, None)
    std = np.sqrt(var)
    std = np.where(std < eps, 1.0, std)

    # Do not standardize binary channels: keep them as-is via mean=0,std=1.
    if binary_indices:
        mean[binary_indices] = 0.0
        std[binary_indices] = 1.0

    return {
        "channel_names": channel_names,
        "binary_channels": [channel_names[i] for i in binary_indices],
        "non_normalized_channels": [channel_names[i] for i in binary_indices],
        "count": count.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "nan_sessions": nan_sessions,
    }


def load_index(path: Union[str, Path]) -> List[Dict[str, int]]:
    """Load a jsonl index file."""
    records: List[Dict[str, int]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


class HDDWindowDataset:
    """Iterable over sliding windows defined by a jsonl index."""

    def __init__(
        self,
        index: Union[str, Path, List[Dict[str, int]]],
        sensor_dir: Union[str, Path],
        label_dir: Optional[Union[str, Path]] = None,
        window: int = DEFAULT_WINDOW,
        normalization: Optional[dict] = None,
        return_label: bool = False,
        label_strategy: str = "end",
        cache_size: int = 2,
        to_tensor: bool = False,
    ) -> None:
        if isinstance(index, (str, Path)):
            self.index = load_index(index)
        else:
            self.index = index
        self.sensor_dir = Path(sensor_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        self.window = window
        self.return_label = return_label and self.label_dir is not None
        self.label_strategy = label_strategy
        self.cache_size = max(1, cache_size)
        self.to_tensor = to_tensor

        if normalization is None:
            self.norm_mean = None
            self.norm_std = None
        else:
            norm = normalization
            if isinstance(normalization, (str, Path)):
                norm = json.loads(Path(normalization).read_text())
            self.norm_mean = np.asarray(norm["mean"], dtype=np.float32)
            self.norm_std = np.asarray(norm["std"], dtype=np.float32)

        self._cache: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        self._cache_order: List[str] = []

    def __len__(self) -> int:
        return len(self.index)

    def _get_session(self, session_id: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if session_id in self._cache:
            return self._cache[session_id]

        x = _load_npy(self.sensor_dir / f"{session_id}.npy")
        y = None
        if self.label_dir is not None and self.label_dir.exists():
            y = _load_npy(self.label_dir / f"{session_id}.npy")

        self._cache[session_id] = (x, y)
        self._cache_order.append(session_id)
        if len(self._cache_order) > self.cache_size:
            drop_id = self._cache_order.pop(0)
            self._cache.pop(drop_id, None)
        return x, y

    def _slice_label(self, y: np.ndarray, t_end: int) -> Optional[int]:
        if y is None:
            return None
        if self.label_strategy == "end":
            return int(y[t_end])
        if self.label_strategy == "center":
            t_center = t_end - self.window // 2
            t_center = max(0, min(len(y) - 1, t_center))
            return int(y[t_center])
        if self.label_strategy == "majority":
            t_start = t_end - (self.window - 1)
            window_labels = y[t_start : t_end + 1]
            values, counts = np.unique(window_labels, return_counts=True)
            return int(values[np.argmax(counts)])
        raise ValueError(f"Unsupported label_strategy: {self.label_strategy}")

    def __getitem__(self, idx: int):
        item = self.index[idx]
        session_id = item["session_id"]
        t_end = item["t_end"]
        x, y = self._get_session(session_id)

        t_start = t_end - (self.window - 1)
        window = np.array(x[t_start : t_end + 1])
        if self.norm_mean is not None and self.norm_std is not None:
            window = window.astype(np.float32, copy=False)
            window = (window - self.norm_mean) / self.norm_std

        label = None
        if self.return_label:
            label = self._slice_label(y, t_end) if y is not None else None

        if self.to_tensor:
            import torch  # lazy import to keep dependency optional

            window = torch.as_tensor(window, dtype=torch.float32)
            if label is not None:
                label = torch.tensor(label, dtype=torch.long)

        metadata = {"session_id": session_id, "t_end": t_end}
        if self.return_label:
            return window, label, metadata
        return window, metadata


def save_splits_and_indices(
    sensor_dir: Union[str, Path],
    label_dir: Union[str, Path],
    artifacts_dir: Union[str, Path] = "artifacts",
    window: int = DEFAULT_WINDOW,
    hop_train: int = DEFAULT_HOP_TRAIN,
    hop_infer: int = DEFAULT_HOP_INFER,
    seed: int = 123,
) -> dict:
    """Convenience helper to run scan, splits, indices, and normalization.

    Naming: sparse -> hop=3 (training stride), dense -> hop=1 (dense inference/segmentation).
    """
    artifacts_dir = Path(artifacts_dir)
    summary = scan_dataset(sensor_dir, label_dir, window=window)
    splits = build_session_splits(summary["per_session_label_counts"].keys(), seed=seed)
    _save_json(splits, artifacts_dir / "splits.json")

    indices = {}
    for split_name, session_ids in splits.items():
        if split_name == "seed":
            continue
        train_mode = build_window_index(session_ids, sensor_dir, window, hop_train)
        infer_mode = build_window_index(session_ids, sensor_dir, window, hop_infer)
        # sparse corresponds to training stride (hop=3); dense is for inference/segmentation (hop=1).
        train_path = artifacts_dir / f"index_{split_name}_sparse.jsonl"
        infer_path = artifacts_dir / f"index_{split_name}_dense.jsonl"
        _save_jsonl(train_mode, train_path)
        _save_jsonl(infer_mode, infer_path)
        indices[split_name] = {
            "sparse": train_path,
            "dense": infer_path,
        }

    normalization = compute_normalization(splits["train"], sensor_dir)
    _save_json(normalization, artifacts_dir / "normalization.json")

    return {"summary": summary, "splits": splits, "indices": indices, "normalization": normalization}


__all__ = [
    "CHANNEL_NAMES",
    "FS_HZ",
    "DEFAULT_WINDOW",
    "DEFAULT_HOP_TRAIN",
    "DEFAULT_HOP_INFER",
    "scan_dataset",
    "build_session_splits",
    "build_window_index",
    "compute_normalization",
    "HDDWindowDataset",
    "load_index",
    "save_splits_and_indices",
]
