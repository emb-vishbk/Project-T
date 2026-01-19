"""Dataset schema constants for the 3 Hz HDD subset."""
from __future__ import annotations

CHANNELS = [
    "accel_pedal_pct",
    "steer_angle_deg",
    "steer_speed",
    "speed",
    "brake_kpa",
    "lturn",
    "rturn",
    "yaw_deg_s",
]

BINARY_CHANNELS = ["lturn", "rturn"]


def channel_index() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(CHANNELS)}


def binary_channel_indices() -> list[int]:
    index = channel_index()
    return [index[name] for name in BINARY_CHANNELS]


def continuous_channel_indices() -> list[int]:
    binary_idx = set(binary_channel_indices())
    return [idx for idx in range(len(CHANNELS)) if idx not in binary_idx]


def continuous_channels() -> list[str]:
    binary = set(BINARY_CHANNELS)
    return [name for name in CHANNELS if name not in binary]
