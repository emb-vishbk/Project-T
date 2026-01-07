"""Lightweight TCN autoencoder for HDD sliding windows."""

from __future__ import annotations

import torch
from torch import nn


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.resample = None
        if in_channels != out_channels:
            self.resample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.act(y)
        y = self.dropout(y)
        res = x if self.resample is None else self.resample(x)
        return self.act(y + res)


class TCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(num_layers):
            dilation = 2**i
            layers.append(
                TemporalBlock(
                    channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            channels = hidden_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Autoencoder(nn.Module):
    """TCN autoencoder for (B, T, C) windows."""

    def __init__(
        self,
        in_channels: int = 8,
        latent_dim: int = 20,
        hidden_channels: int = 32,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.encoder = TCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.enc_pool = nn.AdaptiveAvgPool1d(1)
        self.enc_fc = nn.Linear(hidden_channels, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, hidden_channels)
        self.decoder = TCN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.out_conv = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        x_t = x.transpose(1, 2)
        h = self.encoder(x_t)
        pooled = self.enc_pool(h).squeeze(-1)
        z = self.enc_fc(pooled)
        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, C) -> (B, C, T)
        x_t = x.transpose(1, 2)
        h = self.encoder(x_t)
        pooled = self.enc_pool(h).squeeze(-1)
        z = self.enc_fc(pooled)

        t_len = x_t.shape[-1]
        dec = self.dec_fc(z).unsqueeze(-1).repeat(1, 1, t_len)
        dec = self.decoder(dec)
        recon = self.out_conv(dec).transpose(1, 2)
        return recon, z


__all__ = ["Autoencoder"]
