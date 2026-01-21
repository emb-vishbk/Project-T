"""Encoder registry."""
from __future__ import annotations

from typing import Callable, Dict, Type

ENCODER_REGISTRY: Dict[str, Type] = {}


def register_encoder(name: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        if name in ENCODER_REGISTRY:
            raise ValueError(f"Encoder already registered: {name}")
        ENCODER_REGISTRY[name] = cls
        return cls

    return decorator


def get_encoder(name: str) -> Type:
    if name not in ENCODER_REGISTRY:
        available = ", ".join(sorted(ENCODER_REGISTRY))
        raise KeyError(f"Unknown encoder '{name}'. Available: {available}")
    return ENCODER_REGISTRY[name]
