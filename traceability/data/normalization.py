"""Normalization statistics for continuous channels."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from traceability.data.hdd_io import load_sensors


def compute_normalization_stats(
    sensors_root: str,
    session_ids: Sequence[str],
    continuous_indices: Sequence[int],
    channel_names: Sequence[str],
    binary_channels: Sequence[str],
) -> dict:
    if not session_ids:
        raise ValueError("No session_ids provided for normalization stats.")

    continuous_indices = list(continuous_indices)
    count = 0
    mean = np.zeros(len(continuous_indices), dtype=np.float64)
    m2 = np.zeros(len(continuous_indices), dtype=np.float64)

    for session_id in session_ids:
        sensors = load_sensors(sensors_root, session_id, mmap_mode="r")
        data = np.asarray(sensors[:, continuous_indices], dtype=np.float64)
        if data.size == 0:
            continue

        batch_count = data.shape[0]
        batch_mean = data.mean(axis=0)
        batch_var = data.var(axis=0)

        if count == 0:
            mean = batch_mean
            m2 = batch_var * batch_count
            count = batch_count
            continue

        total_count = count + batch_count
        delta = batch_mean - mean
        mean = mean + delta * batch_count / total_count
        m2 = m2 + batch_var * batch_count + (delta**2) * count * batch_count / total_count
        count = total_count

    if count == 0:
        raise ValueError("No samples found for normalization stats.")

    std = np.sqrt(m2 / count)

    mean_full: list[float | None] = [None] * len(channel_names)
    std_full: list[float | None] = [None] * len(channel_names)
    for idx, channel_idx in enumerate(continuous_indices):
        mean_full[channel_idx] = float(mean[idx])
        std_full[channel_idx] = float(std[idx])

    return {
        "channels": list(channel_names),
        "binary_channels": list(binary_channels),
        "continuous_indices": continuous_indices,
        "mean": mean_full,
        "std": std_full,
        "num_samples": int(count),
        "num_sessions": int(len(session_ids)),
    }
