"""Windowed dataset for HDD sessions."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from torch.utils.data import Dataset

from traceability.data.hdd_io import load_labels, load_sensors
from traceability.data.schema import binary_channel_indices, channel_index, continuous_channel_indices
from traceability.utils.io import read_jsonl


class _SessionCache:
    def __init__(
        self,
        sensors_root: str | Path,
        labels_root: str | Path | None,
        max_sessions: int = 2,
        mmap_mode: str | None = "r",
    ) -> None:
        self._sensors_root = str(sensors_root)
        self._labels_root = str(labels_root) if labels_root is not None else None
        self._max_sessions = max_sessions
        self._mmap_mode = mmap_mode
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray | None]] = OrderedDict()

    def get(self, session_id: str) -> tuple[np.ndarray, np.ndarray | None]:
        if session_id in self._cache:
            self._cache.move_to_end(session_id)
            return self._cache[session_id]

        sensors = load_sensors(self._sensors_root, session_id, mmap_mode=self._mmap_mode)
        labels = None
        if self._labels_root is not None:
            labels = load_labels(self._labels_root, session_id, mmap_mode=self._mmap_mode)

        self._cache[session_id] = (sensors, labels)
        if len(self._cache) > self._max_sessions:
            self._cache.popitem(last=False)
        return sensors, labels


class HDDWindowDataset(Dataset):
    def __init__(
        self,
        sensors_root: str | Path,
        labels_root: str | Path | None,
        index_path: str | Path | None = None,
        index_entries: Iterable[dict] | None = None,
        window_timesteps: int = 1,
        channels: list[str] | None = None,
        normalization_stats: dict | None = None,
        zscore: bool = True,
        clip_sigma: float | None = None,
        return_label: bool = False,
        label_alignment: str = "end",
        cache_size: int = 2,
        mmap_mode: str | None = "r",
    ) -> None:
        if index_entries is None and index_path is None:
            raise ValueError("Provide index_entries or index_path.")
        if index_entries is None:
            index_entries = read_jsonl(index_path)

        self._entries = list(index_entries)
        self._window = int(window_timesteps)
        self._channels = channels or list(channel_index().keys())
        self._return_label = return_label
        self._label_alignment = label_alignment
        self._zscore = zscore
        self._clip_sigma = clip_sigma

        self._binary_indices = binary_channel_indices()
        self._continuous_indices = continuous_channel_indices()

        self._mean = None
        self._std = None
        if normalization_stats is not None:
            mean_full = normalization_stats.get("mean")
            std_full = normalization_stats.get("std")
            if mean_full is not None and std_full is not None:
                mean = np.array([mean_full[i] for i in self._continuous_indices], dtype=np.float32)
                std = np.array([std_full[i] for i in self._continuous_indices], dtype=np.float32)
                std = np.where(std == 0, 1.0, std)
                self._mean = mean
                self._std = std

        self._cache = _SessionCache(
            sensors_root=sensors_root,
            labels_root=labels_root,
            max_sessions=cache_size,
            mmap_mode=mmap_mode,
        )

    @property
    def continuous_indices(self) -> list[int]:
        return self._continuous_indices

    @property
    def binary_indices(self) -> list[int]:
        return self._binary_indices

    def __len__(self) -> int:
        return len(self._entries)

    def _normalize(self, window: np.ndarray) -> np.ndarray:
        if not self._zscore or self._mean is None or self._std is None:
            return window
        window = window.astype(np.float32, copy=True)
        window[:, self._continuous_indices] = (
            window[:, self._continuous_indices] - self._mean
        ) / self._std
        if self._clip_sigma is not None:
            sigma = float(self._clip_sigma)
            window[:, self._continuous_indices] = np.clip(
                window[:, self._continuous_indices], -sigma, sigma
            )
        return window

    def _align_label(self, labels: np.ndarray, t_start: int, t_end: int) -> int:
        if self._label_alignment == "end":
            return int(labels[t_end])
        if self._label_alignment == "center":
            center = t_start + (t_end - t_start) // 2
            return int(labels[center])
        if self._label_alignment == "majority":
            window_labels = labels[t_start : t_end + 1].astype(int)
            counts = np.bincount(window_labels)
            return int(np.argmax(counts))
        raise ValueError(f"Unknown label_alignment: {self._label_alignment}")

    def get_window(self, session_id: str, t_end: int) -> np.ndarray:
        sensors, _ = self._cache.get(session_id)
        t_start = t_end - (self._window - 1)
        if t_start < 0 or t_end >= sensors.shape[0]:
            raise IndexError("Window out of bounds.")
        window = sensors[t_start : t_end + 1, :]
        return self._normalize(window)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._entries[idx]
        session_id = entry["session_id"]
        t_end = int(entry["t_end"])
        sensors, labels = self._cache.get(session_id)
        t_start = t_end - (self._window - 1)
        if t_start < 0 or t_end >= sensors.shape[0]:
            raise IndexError("Window out of bounds.")

        window = self._normalize(sensors[t_start : t_end + 1, :])
        sample: dict[str, Any] = {
            "x": window.astype(np.float32, copy=False),
            "meta": {"session_id": session_id, "t_end": t_end},
        }

        if self._return_label and labels is not None:
            sample["y"] = self._align_label(labels, t_start, t_end)

        return sample
