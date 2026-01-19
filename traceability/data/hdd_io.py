"""IO helpers for the HDD 3 Hz subset stored as NumPy arrays."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _list_npy_stems(root: str | Path) -> set[str]:
    root_path = Path(root)
    return {path.stem for path in root_path.glob("*.npy")}


def list_session_ids(sensors_root: str | Path, labels_root: str | Path) -> list[str]:
    sensor_ids = _list_npy_stems(sensors_root)
    label_ids = _list_npy_stems(labels_root)

    missing_labels = sorted(sensor_ids - label_ids)
    missing_sensors = sorted(label_ids - sensor_ids)
    if missing_labels or missing_sensors:
        raise ValueError(
            "Sensor/label session mismatch: "
            f"missing_labels={missing_labels}, missing_sensors={missing_sensors}"
        )

    return sorted(sensor_ids)


def sensor_path(sensors_root: str | Path, session_id: str) -> Path:
    return Path(sensors_root) / f"{session_id}.npy"


def label_path(labels_root: str | Path, session_id: str) -> Path:
    return Path(labels_root) / f"{session_id}.npy"


def load_sensors(
    sensors_root: str | Path, session_id: str, mmap_mode: str | None = None
) -> np.ndarray:
    return np.load(sensor_path(sensors_root, session_id), mmap_mode=mmap_mode)


def load_labels(
    labels_root: str | Path, session_id: str, mmap_mode: str | None = None
) -> np.ndarray:
    return np.load(label_path(labels_root, session_id), mmap_mode=mmap_mode)
