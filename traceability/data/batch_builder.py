"""Objective-driven batch builder for HDD windows."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()
    return int(digest[:8], 16)


def _apply_augmentations(
    x: np.ndarray,
    rng: np.random.Generator,
    augmentations: list[dict],
    continuous_indices: list[int],
) -> np.ndarray:
    out = x.astype(np.float32, copy=True)
    for aug in augmentations:
        name = aug.get("name")
        kwargs = aug.get("kwargs", {})
        if name == "jitter":
            sigma = float(kwargs.get("sigma", 0.0))
            noise = rng.normal(0.0, sigma, size=out[:, continuous_indices].shape)
            out[:, continuous_indices] += noise
        elif name == "scale":
            scale_min = float(kwargs.get("min", 1.0))
            scale_max = float(kwargs.get("max", 1.0))
            scale = rng.uniform(scale_min, scale_max, size=(1, len(continuous_indices)))
            out[:, continuous_indices] *= scale
        else:
            raise ValueError(f"Unknown augmentation: {name}")
    return out


def _build_mask(
    rng: np.random.Generator,
    shape: tuple[int, int],
    mask_prob: float,
    mask_mode: str,
    binary_indices: list[int],
    mask_binary_channels: bool,
) -> np.ndarray:
    if mask_mode == "time":
        time_mask = rng.random(size=(shape[0],)) < mask_prob
        mask = np.repeat(time_mask[:, None], shape[1], axis=1)
    elif mask_mode == "channel":
        chan_mask = rng.random(size=(shape[1],)) < mask_prob
        mask = np.repeat(chan_mask[None, :], shape[0], axis=0)
    elif mask_mode == "time+channel":
        mask = rng.random(size=shape) < mask_prob
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")

    if not mask_binary_channels and binary_indices:
        mask[:, binary_indices] = False
    return mask


class BatchBuilder:
    def __init__(
        self,
        dataset,
        required_keys: list[str],
        objective_kwargs: dict,
        base_seed: int,
    ) -> None:
        self.dataset = dataset
        self.required_keys = required_keys
        self.objective_kwargs = objective_kwargs
        self.base_seed = base_seed
        self.last_drop_counts: dict[str, int] = {}

    def __call__(self, samples: list[dict]) -> dict | None:
        if not samples:
            return None

        entries = []
        drop_counts: dict[str, int] = {}

        for sample in samples:
            session_id = sample["meta"]["session_id"]
            t_end = int(sample["meta"]["t_end"])
            record: dict[str, Any] = {
                "x": sample["x"],
                "session_id": session_id,
                "t_end": t_end,
            }

            if "x_future" in self.required_keys:
                offset = int(self.objective_kwargs.get("future_offset_timesteps", 1))
                if offset < 1:
                    raise ValueError("future_offset_timesteps must be >= 1.")
                t_end_future = t_end + offset
                try:
                    record["x_future"] = self.dataset.get_window(session_id, t_end_future)
                except IndexError:
                    drop_counts["missing_future"] = drop_counts.get("missing_future", 0) + 1
                    continue

            if "x1" in self.required_keys or "x2" in self.required_keys:
                augmentations = self.objective_kwargs.get("augmentations", [])
                aug_seed = self.objective_kwargs.get("aug_seed", self.base_seed)
                for view_id in ("x1", "x2"):
                    if view_id not in self.required_keys:
                        continue
                    seed = _stable_seed(aug_seed, session_id, t_end, view_id)
                    rng = np.random.default_rng(seed)
                    record[view_id] = _apply_augmentations(
                        record["x"],
                        rng,
                        augmentations,
                        self.dataset.continuous_indices,
                    )

            if any(key in self.required_keys for key in ("x_masked", "mask", "target")):
                mask_prob = float(self.objective_kwargs.get("mask_prob", 0.15))
                mask_mode = str(self.objective_kwargs.get("mask_mode", "time"))
                mask_value = self.objective_kwargs.get("mask_value", 0.0)
                mask_binary = bool(self.objective_kwargs.get("mask_binary_channels", False))
                if mask_value == "learned_token":
                    raise ValueError("mask_value='learned_token' requires encoder support.")
                seed = _stable_seed(self.base_seed, session_id, t_end, "mask")
                rng = np.random.default_rng(seed)
                mask = _build_mask(
                    rng,
                    record["x"].shape,
                    mask_prob,
                    mask_mode,
                    self.dataset.binary_indices,
                    mask_binary,
                )
                x_masked = record["x"].copy()
                x_masked[mask] = float(mask_value)
                if "x_masked" in self.required_keys:
                    record["x_masked"] = x_masked
                if "mask" in self.required_keys:
                    record["mask"] = mask
                if "target" in self.required_keys:
                    record["target"] = record["x"]

            entries.append(record)

        self.last_drop_counts = drop_counts
        if not entries:
            return None

        batch: dict[str, Any] = {"meta": {"session_id": [], "t_end": []}}
        for record in entries:
            batch["meta"]["session_id"].append(record["session_id"])
            batch["meta"]["t_end"].append(record["t_end"])

        for key in self.required_keys:
            array = np.stack([record[key] for record in entries], axis=0)
            tensor = torch.from_numpy(array)
            if key == "mask":
                tensor = tensor.bool()
            batch[key] = tensor

        return batch
