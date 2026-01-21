"""Objective registry."""
from __future__ import annotations

from typing import Callable, Dict, Type

OBJECTIVE_REGISTRY: Dict[str, Type] = {}


def register_objective(name: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        if name in OBJECTIVE_REGISTRY:
            raise ValueError(f"Objective already registered: {name}")
        OBJECTIVE_REGISTRY[name] = cls
        return cls

    return decorator


def get_objective(name: str) -> Type:
    if name not in OBJECTIVE_REGISTRY:
        available = ", ".join(sorted(OBJECTIVE_REGISTRY))
        raise KeyError(f"Unknown objective '{name}'. Available: {available}")
    return OBJECTIVE_REGISTRY[name]
