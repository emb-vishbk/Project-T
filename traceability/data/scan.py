"""Dataset scan and validation for the HDD 3 Hz subset."""
from __future__ import annotations

from typing import Any

import numpy as np

from traceability.data.hdd_io import list_session_ids, load_labels, load_sensors


def scan_dataset(
    sensors_root: str,
    labels_root: str,
    fs: float,
    expected_channels: int = 8,
) -> tuple[dict[str, Any], list[str], dict[str, int]]:
    session_ids = list_session_ids(sensors_root, labels_root)

    session_lengths: dict[str, int] = {}
    session_label_diversity: dict[str, int] = {}
    label_counts: dict[int, int] = {}
    label_inventory: set[int] = set()
    sensor_dtypes: set[str] = set()
    label_dtypes: set[str] = set()

    for session_id in session_ids:
        sensors = load_sensors(sensors_root, session_id, mmap_mode="r")
        labels = load_labels(labels_root, session_id, mmap_mode="r")

        if sensors.ndim != 2 or sensors.shape[1] != expected_channels:
            raise ValueError(
                f"Unexpected sensor shape for {session_id}: {sensors.shape}"
            )
        if labels.ndim != 1:
            raise ValueError(f"Unexpected label shape for {session_id}: {labels.shape}")
        if sensors.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Length mismatch for {session_id}: X={sensors.shape[0]} y={labels.shape[0]}"
            )

        length = int(sensors.shape[0])
        session_lengths[session_id] = length
        sensor_dtypes.add(str(sensors.dtype))
        label_dtypes.add(str(labels.dtype))

        labels_np = np.asarray(labels)
        unique, counts = np.unique(labels_np, return_counts=True)
        session_label_diversity[session_id] = int(unique.size)
        for label, count in zip(unique, counts):
            label_int = int(label)
            label_counts[label_int] = label_counts.get(label_int, 0) + int(count)
            label_inventory.add(label_int)

    lengths = list(session_lengths.values())
    diversity_values = list(session_label_diversity.values())
    total_timesteps = int(sum(lengths))
    duration_minutes = float(total_timesteps / fs / 60.0)

    summary = {
        "num_sessions": len(session_ids),
        "fs_hz": fs,
        "total_timesteps": total_timesteps,
        "total_duration_minutes": duration_minutes,
        "length_stats": {
            "min": int(min(lengths)) if lengths else 0,
            "max": int(max(lengths)) if lengths else 0,
            "mean": float(np.mean(lengths)) if lengths else 0.0,
        },
        "label_diversity_stats": {
            "min": int(min(diversity_values)) if diversity_values else 0,
            "max": int(max(diversity_values)) if diversity_values else 0,
            "mean": float(np.mean(diversity_values)) if diversity_values else 0.0,
        },
        "sensor_dtypes": sorted(sensor_dtypes),
        "label_dtypes": sorted(label_dtypes),
        "label_inventory": sorted(label_inventory),
        "label_counts": {str(label): count for label, count in sorted(label_counts.items())},
        "session_lengths": session_lengths,
        "session_label_diversity": session_label_diversity,
    }

    return summary, session_ids, session_lengths
