"""Window index generation utilities."""
from __future__ import annotations

from typing import Sequence

from traceability.data.hdd_io import load_sensors


def build_window_index(
    sensors_root: str,
    session_ids: Sequence[str],
    window_timesteps: int,
    hop_timesteps: int,
    sort_sessions: bool = True,
) -> tuple[list[dict], dict]:
    if window_timesteps <= 0:
        raise ValueError("window_timesteps must be >= 1")
    if hop_timesteps <= 0:
        raise ValueError("hop_timesteps must be >= 1")

    ordered_sessions = sorted(session_ids) if sort_sessions else list(session_ids)
    entries: list[dict] = []
    per_session_counts: dict[str, int] = {}
    skipped_sessions: list[str] = []

    for session_id in ordered_sessions:
        sensors = load_sensors(sensors_root, session_id, mmap_mode="r")
        length = int(sensors.shape[0])
        if length < window_timesteps:
            skipped_sessions.append(session_id)
            per_session_counts[session_id] = 0
            continue

        start = window_timesteps - 1
        count = 0
        for t_end in range(start, length, hop_timesteps):
            entries.append({"session_id": session_id, "t_end": int(t_end)})
            count += 1
        per_session_counts[session_id] = count

    summary = {
        "num_sessions": len(ordered_sessions),
        "num_windows": len(entries),
        "window_timesteps": int(window_timesteps),
        "hop_timesteps": int(hop_timesteps),
        "skipped_sessions": skipped_sessions,
        "per_session_counts": per_session_counts,
    }

    return entries, summary
