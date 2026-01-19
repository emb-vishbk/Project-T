"""Session-level splitting utilities."""
from __future__ import annotations

from typing import Sequence

from traceability.utils.seed import rng_from_seed


def split_sessions(
    session_ids: Sequence[str],
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, list[str]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    ids = list(session_ids)
    rng = rng_from_seed(seed)
    rng.shuffle(ids)

    n_total = len(ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_ids = sorted(ids[:n_train])
    val_ids = sorted(ids[n_train : n_train + n_val])
    test_ids = sorted(ids[n_train + n_val :])

    return {"train": train_ids, "val": val_ids, "test": test_ids}
