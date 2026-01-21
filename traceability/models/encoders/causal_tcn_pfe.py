"""Causal TCN encoder with late-time pooling and future prediction head."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from traceability.models.encoders.registry import register_encoder


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self._pad = (kernel_size - 1) * dilation
        self._conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self._pad, 0))
        return self._conv(x)


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self._conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self._conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self._dropout = nn.Dropout(dropout)
        self._act = nn.ReLU()
        self._residual = None
        if in_channels != out_channels:
            self._residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._conv1(x)
        out = self._act(out)
        out = self._dropout(out)
        out = self._conv2(out)
        out = self._act(out)
        out = self._dropout(out)
        residual = x if self._residual is None else self._residual(x)
        return out + residual


@register_encoder("tcn_causal_pfe")
class CausalTCNPFE(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        embedding_dim: int = 64,
        tcn_channels: int = 64,
        tcn_depth: int = 4,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.1,
        pooling: dict[str, Any] | None = None,
        future_mlp_hidden: int | None = None,
    ) -> None:
        super().__init__()
        if tcn_depth < 1:
            raise ValueError("tcn_depth must be >= 1.")
        if kernel_size < 2:
            raise ValueError("kernel_size must be >= 2 for causal conv.")

        pooling = pooling or {"type": "last_k", "k": 1}
        self._pool_type = str(pooling.get("type", "last_k"))
        self._pool_k = int(pooling.get("k", 1))
        if self._pool_k < 1:
            raise ValueError("pooling.k must be >= 1.")
        if self._pool_type not in {"last_k", "attn"}:
            raise ValueError(f"Unsupported pooling type: {self._pool_type}")

        layers = []
        in_channels = input_dim
        for idx in range(tcn_depth):
            dilation = dilation_base ** idx
            layers.append(
                TCNBlock(
                    in_channels=in_channels,
                    out_channels=tcn_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = tcn_channels
        self._tcn = nn.Sequential(*layers)
        self._proj = nn.Conv1d(in_channels, embedding_dim, kernel_size=1)

        self._attn = None
        if self._pool_type == "attn":
            self._attn = nn.Linear(embedding_dim, 1)

        hidden_dim = future_mlp_hidden or embedding_dim
        self._future_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def _encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError("Expected input shape (B, W, C) or (W, C).")
        x = x.transpose(1, 2)
        features = self._tcn(x)
        return self._proj(features)

    def _pool(self, features: torch.Tensor) -> torch.Tensor:
        total_steps = features.shape[-1]
        k = min(self._pool_k, total_steps)
        if self._pool_type == "last_k":
            return features[..., -k:].mean(dim=-1)

        features = features[..., -k:]
        steps = features.transpose(1, 2)
        scores = self._attn(steps).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(steps * weights.unsqueeze(-1), dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 2
        features = self._encode_sequence(x)
        pooled = self._pool(features)
        return pooled.squeeze(0) if squeeze else pooled

    def predict_future(self, z: torch.Tensor) -> torch.Tensor:
        squeeze = z.dim() == 1
        if squeeze:
            z = z.unsqueeze(0)
        z_hat = self._future_head(z)
        return z_hat.squeeze(0) if squeeze else z_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)
