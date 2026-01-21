"""Predictive future objective in latent space."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from traceability.objectives.registry import register_objective


@register_objective("predictive_future_latent")
class PredictiveFutureLatent:
    REQUIRED_BATCH_KEYS = ["x", "x_future"]
    REQUIRED_ENCODER_METHODS = ["encode", "predict_future"]

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        cosine_weight: float = 0.0,
        **kwargs,
    ) -> None:
        self.loss_type = str(loss_type).lower()
        self.cosine_weight = float(cosine_weight)
        self.kwargs = kwargs

    def _latent_loss(self, z_hat: torch.Tensor, z_f: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(z_hat, z_f)
        if self.loss_type == "mse":
            return F.mse_loss(z_hat, z_f)
        raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    def compute_loss(self, encoder, batch: dict) -> tuple[torch.Tensor, dict]:
        x = batch["x"]
        x_future = batch["x_future"]

        z = encoder.encode(x)
        z_hat = encoder.predict_future(z)
        with torch.no_grad():
            z_f = encoder.encode(x_future).detach()

        recon_loss = self._latent_loss(z_hat, z_f)
        loss = recon_loss
        logs = {"loss_recon": float(recon_loss.item())}

        if self.cosine_weight > 0.0:
            cos = 1.0 - F.cosine_similarity(z_hat, z_f, dim=-1).mean()
            loss = loss + self.cosine_weight * cos
            logs["loss_cosine"] = float(cos.item())

        logs["loss_total"] = float(loss.item())
        return loss, logs
