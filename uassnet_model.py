"""Shared UASS-Net architecture and checkpoint helpers.

Training and inference import the same model definition from this module so
that a checkpoint cannot silently drift between the two workflows.  The
architecture and module attribute names preserve the released checkpoint key
layout.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_channel_argument(
    canonical_name: str,
    canonical_value: int | None,
    legacy_name: str,
    legacy_value: int | None,
    *,
    default: int | None = None,
) -> int:
    """Resolve current and legacy constructor keyword names."""

    if canonical_value is not None and legacy_value is not None:
        if canonical_value != legacy_value:
            raise TypeError(
                f"Conflicting values for {canonical_name!r} and its legacy "
                f"alias {legacy_name!r}."
            )
        return canonical_value
    value = canonical_value if canonical_value is not None else legacy_value
    if value is None:
        if default is None:
            raise TypeError(f"Missing required argument: {canonical_name!r}.")
        value = default
    return value


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        *,
        in_ch: int | None = None,
        out_ch: int | None = None,
    ) -> None:
        super().__init__()
        in_channels = _resolve_channel_argument(
            "in_channels", in_channels, "in_ch", in_ch
        )
        out_channels = _resolve_channel_argument(
            "out_channels", out_channels, "out_ch", out_ch
        )
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.layers(inputs)


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        *,
        in_ch: int | None = None,
        out_ch: int | None = None,
    ) -> None:
        super().__init__()
        in_channels = _resolve_channel_argument(
            "in_channels", in_channels, "in_ch", in_ch
        )
        out_channels = _resolve_channel_argument(
            "out_channels", out_channels, "out_ch", out_ch
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=6,
                dilation=6,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=12,
                dilation=12,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=18,
                dilation=18,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        branches = (self.b1(inputs), self.b2(inputs), self.b3(inputs), self.b4(inputs))
        return self.fuse(torch.cat(branches, dim=1))


class DecoderBlock(nn.Module):
    def __init__(
        self,
        up_channels: int | None = None,
        skip_channels: int | None = None,
        out_channels: int | None = None,
        dropout: float = 0.2,
        *,
        up_ch: int | None = None,
        skip_ch: int | None = None,
        out_ch: int | None = None,
    ) -> None:
        super().__init__()
        up_channels = _resolve_channel_argument(
            "up_channels", up_channels, "up_ch", up_ch
        )
        skip_channels = _resolve_channel_argument(
            "skip_channels", skip_channels, "skip_ch", skip_ch
        )
        out_channels = _resolve_channel_argument(
            "out_channels", out_channels, "out_ch", out_ch
        )
        self.se = SEBlock(skip_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.conv = ConvBlock(up_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Nearest interpolation is part of the checkpoint's reference forward definition.
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="nearest")
        inputs = torch.cat([self.se(skip), inputs], dim=1)
        return self.conv(self.dropout(inputs))


class UASSNet(nn.Module):
    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        base_channels: int | None = None,
        dropout: float = 0.2,
        *,
        in_ch: int | None = None,
        out_ch: int | None = None,
        base: int | None = None,
    ) -> None:
        """Build UASS-Net, accepting both current and legacy keyword names."""

        super().__init__()
        in_channels = _resolve_channel_argument(
            "in_channels", in_channels, "in_ch", in_ch, default=1
        )
        out_channels = _resolve_channel_argument(
            "out_channels", out_channels, "out_ch", out_ch, default=1
        )
        base_channels = _resolve_channel_argument(
            "base_channels", base_channels, "base", base, default=16
        )
        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 32,
        )
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.enc5 = ConvBlock(c4, c5)
        self.pool = nn.MaxPool2d(2)
        self.aspp = ASPP(c5, c4)
        self.dec4 = DecoderBlock(c4, c4, c4, dropout)
        self.dec3 = DecoderBlock(c4, c3, c3, dropout)
        self.dec2 = DecoderBlock(c3, c2, c2, dropout)
        self.dec1 = DecoderBlock(c2, c1, c1, dropout)
        self.out_conv = nn.Conv2d(c1, out_channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder1 = self.enc1(inputs)
        encoder2 = self.enc2(self.pool(encoder1))
        encoder3 = self.enc3(self.pool(encoder2))
        encoder4 = self.enc4(self.pool(encoder3))
        encoder5 = self.enc5(self.pool(encoder4))
        decoded = self.aspp(encoder5)
        decoded = self.dec4(decoded, encoder4)
        decoded = self.dec3(decoded, encoder3)
        decoded = self.dec2(decoded, encoder2)
        decoded = self.dec1(decoded, encoder1)
        return self.out_conv(decoded)


def _looks_like_state_dict(candidate: Any) -> bool:
    return (
        isinstance(candidate, Mapping)
        and len(candidate) > 0
        and all(isinstance(key, str) for key in candidate)
        and all(torch.is_tensor(value) for value in candidate.values())
    )


def extract_model_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Extract weights from a raw or commonly wrapped PyTorch checkpoint.

    Supported wrapper keys are ``model``, ``model_state_dict``, and
    ``state_dict``.  A uniform ``module.`` prefix produced by DataParallel is
    removed without changing the underlying UASS-Net parameter names.
    """

    state_dict: Any = checkpoint
    if not _looks_like_state_dict(state_dict) and isinstance(checkpoint, Mapping):
        for key in ("model", "model_state_dict", "state_dict"):
            candidate = checkpoint.get(key)
            if _looks_like_state_dict(candidate):
                state_dict = candidate
                break

    if not _looks_like_state_dict(state_dict):
        raise ValueError(
            "Unsupported checkpoint format. Expected a raw state_dict or a "
            "mapping containing 'model', 'model_state_dict', or 'state_dict'."
        )

    if all(key.startswith("module.") for key in state_dict):
        return {key[7:]: value for key, value in state_dict.items()}
    return state_dict


def load_torch_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device,
) -> Any:
    """Load a checkpoint while supporting PyTorch versions before 2.0."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
        try:
            return torch.load(
                checkpoint_path,
                map_location=map_location,
                weights_only=True,
            )
        except TypeError:
            return torch.load(checkpoint_path, map_location=map_location)


__all__ = [
    "ASPP",
    "ConvBlock",
    "DecoderBlock",
    "SEBlock",
    "UASSNet",
    "extract_model_state_dict",
    "load_torch_checkpoint",
]
