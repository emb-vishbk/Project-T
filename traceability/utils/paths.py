"""Path helpers for all filesystem access in the project."""
from __future__ import annotations

from pathlib import Path


def get_artifacts_root(artifacts_root: str | Path) -> Path:
    return Path(artifacts_root)


def run_dir(
    artifacts_root: str | Path,
    encoder_name: str,
    objective_name: str,
    run_id: str,
) -> Path:
    return get_artifacts_root(artifacts_root) / "runs" / encoder_name / objective_name / run_id


def tb_dir(run_dir_path: str | Path) -> Path:
    return Path(run_dir_path) / "tb"


def embeddings_dir(artifacts_root: str | Path, split: str) -> Path:
    return get_artifacts_root(artifacts_root) / "embeddings" / split


def embeddings_file(artifacts_root: str | Path, split: str, session_id: str) -> Path:
    return embeddings_dir(artifacts_root, split) / f"{session_id}.npy"


def t_end_file(artifacts_root: str | Path, split: str, session_id: str) -> Path:
    return embeddings_dir(artifacts_root, split) / f"{session_id}_t_end.npy"


def get_data_root(data_root: str | Path) -> Path:
    return Path(data_root)


def raw_root(data_root: str | Path) -> Path:
    return get_data_root(data_root) / "raw"


def sensors_dir(data_root: str | Path) -> Path:
    return raw_root(data_root) / "20200710_sensors" / "sensor"


def labels_dir(data_root: str | Path) -> Path:
    return raw_root(data_root) / "20200710_labels" / "target"


def meta_dir(data_root: str | Path) -> Path:
    return get_data_root(data_root) / "meta"


def index_dir(data_root: str | Path) -> Path:
    return get_data_root(data_root) / "index"


def splits_file(data_root: str | Path) -> Path:
    return meta_dir(data_root) / "splits.json"


def normalization_file(data_root: str | Path) -> Path:
    return meta_dir(data_root) / "normalization.json"


def dataset_summary_file(data_root: str | Path) -> Path:
    return meta_dir(data_root) / "dataset_summary.json"


def index_file(data_root: str | Path, index_family: str, split: str) -> Path:
    return index_dir(data_root) / f"index_{index_family}_{split}.jsonl"
