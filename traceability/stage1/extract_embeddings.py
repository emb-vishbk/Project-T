"""Embedding extraction for Stage 1 encoders."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from traceability.data.dataset import HDDWindowDataset
from traceability.data.indexing import build_window_index
from traceability.models.encoders import get_encoder
from traceability.utils import paths
from traceability.utils.io import read_json, read_jsonl


def _select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_encoder(run_dir: Path, config: dict, device: torch.device):
    encoder_cls = get_encoder(config["encoder"]["name"])
    encoder = encoder_cls(**config["encoder"].get("kwargs", {}))

    weights_best = paths.weights_best_file(run_dir)
    weights_last = paths.weights_last_file(run_dir)
    weights_path = weights_best if weights_best.exists() else weights_last
    if not weights_path.exists():
        raise FileNotFoundError(f"Encoder weights not found in {run_dir}")
    state = torch.load(weights_path, map_location=device)
    encoder.load_state_dict(state)
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


def _resolve_data_paths(config: dict, data_root_override: str | None) -> dict[str, str]:
    data_root = data_root_override or config["io"]["data_root"]
    data_paths = config.get("data", {}).get("paths", {})
    sensors_root = data_paths.get("sensors_root") or str(paths.sensors_dir(data_root))
    labels_root = data_paths.get("labels_root") or str(paths.labels_dir(data_root))
    return {"data_root": data_root, "sensors_root": sensors_root, "labels_root": labels_root}


def _load_inference_index(
    data_root: str,
    sensors_root: str,
    split: str,
    window_timesteps: int,
    hop_infer_timesteps: int,
) -> list[dict]:
    index_path = paths.index_file(data_root, "inference", split)
    if Path(index_path).exists():
        return read_jsonl(index_path)

    splits = read_json(paths.splits_file(data_root))
    session_ids = splits["splits"][split]
    entries, _ = build_window_index(
        sensors_root,
        session_ids,
        window_timesteps=window_timesteps,
        hop_timesteps=hop_infer_timesteps,
    )
    return entries


def _group_entries(entries: Iterable[dict]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for entry in entries:
        grouped[entry["session_id"]].append(int(entry["t_end"]))
    return grouped


def extract_embeddings(
    run_dir: str | Path,
    splits: list[str],
    data_root_override: str | None = None,
    artifacts_root_override: str | None = None,
) -> None:
    run_dir = Path(run_dir)
    config = read_json(paths.config_file(run_dir))

    data_paths = _resolve_data_paths(config, data_root_override)
    normalization = read_json(paths.normalization_file(data_paths["data_root"]))

    device = _select_device(config["training"]["device"])
    encoder = _load_encoder(run_dir, config, device)

    dataset_args = dict(
        sensors_root=data_paths["sensors_root"],
        labels_root=None,
        window_timesteps=config["data"]["window_timesteps"],
        channels=config["data"]["channels"],
        normalization_stats=normalization,
        zscore=config["data"]["normalization"].get("zscore", True),
        clip_sigma=config["data"]["normalization"].get("clip_sigma"),
        return_label=False,
    )

    batch_size = int(config["training"]["batch_size"])

    for split in splits:
        index_entries = _load_inference_index(
            data_paths["data_root"],
            data_paths["sensors_root"],
            split,
            window_timesteps=config["data"]["window_timesteps"],
            hop_infer_timesteps=config["data"]["hop_infer_timesteps"],
        )
        session_to_tend = _group_entries(index_entries)
        dataset = HDDWindowDataset(index_entries=[], **dataset_args)

        output_run_dir = run_dir
        if artifacts_root_override:
            output_run_dir = paths.run_dir(
                artifacts_root_override,
                config["encoder"]["name"],
                config["objective"]["name"],
                config["run"]["run_id"],
            )

        for session_id, t_end_list in session_to_tend.items():
            t_end_list = sorted(t_end_list)
            embeddings = []
            for start in range(0, len(t_end_list), batch_size):
                batch_t_end = t_end_list[start : start + batch_size]
                windows = [dataset.get_window(session_id, t_end) for t_end in batch_t_end]
                x = torch.from_numpy(np.stack(windows, axis=0)).to(device)
                with torch.no_grad():
                    z = encoder.encode(x)
                embeddings.append(z.detach().cpu().numpy())

            z_session = np.concatenate(embeddings, axis=0)
            t_end_arr = np.array(t_end_list, dtype=np.int64)

            out_dir = paths.embeddings_dir(output_run_dir, split)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(paths.embeddings_file(output_run_dir, split, session_id), z_session)
            np.save(paths.t_end_file(output_run_dir, split, session_id), t_end_arr)
