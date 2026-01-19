"""Deterministic seeding helpers."""
from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
