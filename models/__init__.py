"""Model modules for HDD stage 1."""

from .ae_tcn import Autoencoder

_MODEL_REGISTRY = {
    "tcn_ae": Autoencoder,
}


def build_model(model_name: str, **kwargs):
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name: {model_name}")
    return _MODEL_REGISTRY[model_name](**kwargs)


__all__ = ["Autoencoder", "build_model"]
