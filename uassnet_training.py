#!/usr/bin/env python3
"""
Train UASS-Net for 2D thrust-fault probability prediction.

The script implements the model-training part of the manuscript:
synthetic pretraining followed by optional transfer learning on sparse real labels.

Run commands from the repository root. Synthetic pretraining uses the generated
synthetic dataset by default:

    python uassnet_training.py --mode synthetic

For transfer learning, select the generated real-label dataset explicitly:

    python uassnet_training.py \
        --mode transfer \
        --data-root outputs/datasets/real_labels \
        --pretrained outputs/training/uassnet_synthetic_best.pth

All seismic patches, labels, prediction arrays, and network spatial dimensions
use ``(x, z)`` order.  The training tensor layout is therefore
``(batch, channel, x, z)``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


CANONICAL_ARRAY_AXIS_ORDER = ("x", "z")


@dataclass
class TrainConfig:
    data_root: str
    out_dir: str = "outputs/training"
    file_ext: str = ".npy"
    array_axis_order: Tuple[str, str] = CANONICAL_ARRAY_AXIS_ORDER

    patch_size: int = 256
    batch_size: int = 16
    epochs: int = 66
    lr: float = 1e-3
    step_size: int = 20
    gamma: float = 0.1
    weight_decay: float = 0.0

    base_channels: int = 16
    dropout: float = 0.2

    wbce_weight: float = 0.9
    dice_weight: float = 0.1
    eps: float = 1e-7

    random_flip: bool = True
    seed: int = 2026
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def minmax_then_zscore(image: np.ndarray) -> np.ndarray:
    """Min-max normalize a seismic patch, then standardize it by z-score."""

    image = np.asarray(image, dtype=np.float32)
    image_min = float(np.min(image))
    image_max = float(np.max(image))
    if not image_max > image_min:
        raise ValueError("Cannot normalize a constant seismic patch.")

    normalized = (image - image_min) / (image_max - image_min)
    patch_mean = float(np.mean(normalized))
    patch_std = float(np.std(normalized))
    if not patch_std > 0.0:
        raise ValueError("Cannot standardize a constant seismic patch.")

    return ((normalized - patch_mean) / patch_std).astype(np.float32)


def read_dat_patch(
    path: str | Path,
    *,
    dtype: Any,
    patch_size: int,
) -> np.ndarray:
    """Read a headerless C-order (x, z) square patch after byte validation."""

    path = Path(path)
    data_type = np.dtype(dtype)
    expected_values = int(patch_size) ** 2
    expected_bytes = expected_values * data_type.itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{path} has {actual_bytes} bytes; expected {expected_bytes} bytes "
            f"for a {patch_size}x{patch_size} {data_type.name} patch."
        )
    return np.fromfile(path, dtype=data_type).reshape(patch_size, patch_size)


def validate_synthetic_dataset_axis_contract(root: str | Path) -> None:
    """Reject legacy synthetic datasets that were stored in (z, x) order."""
    root = Path(root)
    config_path = root / "config.json"
    if not config_path.is_file():
        return

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read dataset configuration {config_path}: {exc}") from exc

    synthetic_keys = {"height", "width", "num_train", "num_val"}
    if not isinstance(config, dict) or not synthetic_keys.issubset(config):
        return

    axis_order = config.get("array_axis_order")
    if axis_order != list(CANONICAL_ARRAY_AXIS_ORDER):
        declared = "missing (legacy datasets used (z, x))" if axis_order is None else repr(axis_order)
        raise ValueError(
            f"Synthetic dataset {root} has array_axis_order={declared}; expected ['x', 'z']. "
            "Regenerate it with the current synthetic_thrust_data_generation.py "
            "in a new or empty directory before training."
        )


def flip_x(array: np.ndarray) -> np.ndarray:
    """Reverse physical x (axis 0) for an array in canonical (x, z) order."""
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D (x, z) array, received shape {array.shape}.")
    return np.flip(array, axis=0).copy()


class FaultPatchDataset(Dataset):
    """Exact-stem paired seismic/label patches in canonical (x, z) order."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        file_ext: str = ".npy",
        patch_size: int = 256,
        random_flip: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.file_ext = file_ext
        self.patch_size = patch_size
        self.random_flip = random_flip

        supported_extensions = {".npy", ".dat"}
        if self.file_ext not in supported_extensions:
            raise ValueError(
                f"Unsupported dataset file_ext={self.file_ext!r}; expected one of "
                f"{sorted(supported_extensions)}."
            )

        validate_synthetic_dataset_axis_contract(self.root)

        self.image_dir = self.root / split / "seismic"
        self.label_dir = self.root / split / "labels"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Missing label directory: {self.label_dir}")

        image_by_stem = {
            path.stem: path
            for path in self.image_dir.glob(f"*{file_ext}")
            if path.is_file()
        }
        label_by_stem = {
            path.stem: path
            for path in self.label_dir.glob(f"*{file_ext}")
            if path.is_file()
        }

        if not image_by_stem:
            raise RuntimeError(f"No image files found in {self.image_dir}")

        image_stems = set(image_by_stem)
        label_stems = set(label_by_stem)
        if image_stems != label_stems:
            missing_labels = sorted(image_stems - label_stems)
            missing_images = sorted(label_stems - image_stems)

            def preview(stems: List[str]) -> str:
                shown = ", ".join(stems[:5])
                return shown + (", ..." if len(stems) > 5 else "")

            details = []
            if missing_labels:
                details.append(f"missing labels for stems [{preview(missing_labels)}]")
            if missing_images:
                details.append(f"missing seismic for stems [{preview(missing_images)}]")
            raise RuntimeError(
                f"Image/label stem mismatch in split {split!r}: " + "; ".join(details)
            )

        paired_stems = sorted(image_stems)
        self.image_files = [image_by_stem[stem] for stem in paired_stems]
        self.label_files = [label_by_stem[stem] for stem in paired_stems]

    def __len__(self) -> int:
        return len(self.image_files)

    def _read_array(self, path: Path, is_label: bool) -> np.ndarray:
        if path.suffix == ".npy":
            arr = np.load(path)
        elif path.suffix == ".dat":
            dtype = np.uint8 if is_label else np.float32
            arr = read_dat_patch(path, dtype=dtype, patch_size=self.patch_size)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        arr = np.squeeze(arr).astype(np.float32)

        if arr.shape != (self.patch_size, self.patch_size):
            raise ValueError(f"{path} has shape {arr.shape}, expected {(self.patch_size, self.patch_size)}")

        if is_label:
            arr = (arr > 0.5).astype(np.float32)
        else:
            arr = minmax_then_zscore(arr)

        return arr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self._read_array(self.image_files[idx], is_label=False)
        label = self._read_array(self.label_files[idx], is_label=True)

        if self.random_flip and np.random.rand() < 0.5:
            image = flip_x(image)
            label = flip_x(label)

        image = torch.from_numpy(image[None]).float()
        label = torch.from_numpy(label[None]).float()
        return image, label


def make_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, FaultPatchDataset]:
    train_set = FaultPatchDataset(
        cfg.data_root,
        split="train",
        file_ext=cfg.file_ext,
        patch_size=cfg.patch_size,
        random_flip=cfg.random_flip,
    )
    val_set = FaultPatchDataset(
        cfg.data_root,
        split="val",
        file_ext=cfg.file_ext,
        patch_size=cfg.patch_size,
        random_flip=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, train_set


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.layers(x)


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 4, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.fuse(x)


class DecoderBlock(nn.Module):
    def __init__(self, up_ch: int, skip_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.se = SEBlock(skip_ch)
        self.dropout = nn.Dropout2d(dropout)
        self.conv = ConvBlock(up_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="nearest"
        )
        skip = self.se(skip)
        x = torch.cat([skip, x], dim=1)
        x = self.dropout(x)
        return self.conv(x)


class UASSNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 16, dropout: float = 0.2):
        super().__init__()

        c1 = base
        c2 = base * 2
        c3 = base * 4
        c4 = base * 8
        c5 = base * 32

        self.enc1 = ConvBlock(in_ch, c1)
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

        self.out_conv = nn.Conv2d(c1, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        x = self.aspp(e5)
        x = self.dec4(x, e4)
        x = self.dec3(x, e3)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)
        return self.out_conv(x)


def compute_class_weights(dataset: FaultPatchDataset) -> Tuple[float, float]:
    pos = 0.0
    total = 0.0

    for path in dataset.label_files:
        label = dataset._read_array(path, is_label=True)
        pos += float(label.sum())
        total += float(label.size)

    neg = total - pos
    if pos <= 0 or neg <= 0:
        return 1.0, 1.0

    w_pos = neg / total
    w_neg = pos / total
    return w_pos, w_neg


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    w_pos: float,
    w_neg: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.where(targets > 0.5, torch.as_tensor(w_pos, device=targets.device), torch.as_tensor(w_neg, device=targets.device))
    return (weights * bce).mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    w_pos: float,
    w_neg: float,
    cfg: TrainConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    wbce = weighted_bce_with_logits(logits, targets, w_pos, w_neg)
    dice = dice_loss(logits, targets, eps=cfg.eps)
    loss = cfg.wbce_weight * wbce + cfg.dice_weight * dice

    return loss, {
        "loss": float(loss.detach().cpu()),
        "wbce": float(wbce.detach().cpu()),
        "dice": float(dice.detach().cpu()),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    cfg: TrainConfig,
    w_pos: float,
    w_neg: float,
    optimizer: Optional[optim.Optimizer] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "wbce": 0.0, "dice": 0.0}
    n = 0

    for images, labels in loader:
        images = images.to(cfg.device, non_blocking=True)
        labels = labels.to(cfg.device, non_blocking=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss, stats = combined_loss(logits, labels, w_pos, w_neg, cfg)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        for k in totals:
            totals[k] += stats[k]
        n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    history: List[Dict],
    cfg: TrainConfig,
    class_weights: Tuple[float, float],
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "config": asdict(cfg),
            "class_weights": class_weights,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def _looks_like_state_dict(candidate: Any) -> bool:
    """Return True for a non-empty mapping of parameter names to tensors."""

    return (
        isinstance(candidate, Mapping)
        and len(candidate) > 0
        and all(isinstance(key, str) for key in candidate)
        and all(isinstance(value, torch.Tensor) for value in candidate.values())
    )


def _extract_model_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Extract weights from raw or commonly wrapped PyTorch checkpoints."""

    if _looks_like_state_dict(checkpoint):
        return checkpoint

    if isinstance(checkpoint, Mapping):
        for key in ("model", "model_state_dict", "state_dict"):
            candidate = checkpoint.get(key)
            if _looks_like_state_dict(candidate):
                return candidate

    raise ValueError(
        "Unsupported checkpoint format. Expected a raw state_dict or a mapping "
        "containing 'model', 'model_state_dict', or 'state_dict'."
    )


def load_model_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: str,
    strict: bool = True,
) -> Any:
    """Load model weights from a raw state dict or a wrapped checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path.resolve()}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # PyTorch versions before the weights_only argument remain supported.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state = _extract_model_state_dict(checkpoint)
    model.load_state_dict(state, strict=strict)
    return checkpoint


def freeze_encoder(model: UASSNet) -> None:
    for module in [model.enc1, model.enc2, model.enc3, model.enc4, model.enc5]:
        for p in module.parameters():
            p.requires_grad = False


def build_model(cfg: TrainConfig) -> UASSNet:
    return UASSNet(
        in_ch=1,
        out_ch=1,
        base=cfg.base_channels,
        dropout=cfg.dropout,
    ).to(cfg.device)


def train(cfg: TrainConfig, checkpoint_name: str, pretrained: Optional[str] = None, freeze: bool = False) -> Path:
    if freeze and not pretrained:
        raise ValueError("freeze=True requires a pretrained checkpoint.")

    set_seed(cfg.seed)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, train_set = make_loaders(cfg)
    w_pos, w_neg = compute_class_weights(train_set)

    model = build_model(cfg)

    if pretrained:
        load_model_weights(model, pretrained, cfg.device, strict=True)

    if freeze:
        freeze_encoder(model)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    history: List[Dict] = []
    best_val_loss = math.inf
    best_path = Path(cfg.out_dir) / checkpoint_name

    print(f"Training samples: {len(train_set)}")
    print(f"Class weights: w_pos={w_pos:.6f}, w_neg={w_neg:.6f}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, cfg.epochs + 1):
        train_stats = run_epoch(model, train_loader, cfg, w_pos, w_neg, optimizer)
        val_stats = run_epoch(model, val_loader, cfg, w_pos, w_neg, optimizer=None)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["loss"],
            "train_wbce": train_stats["wbce"],
            "train_dice": train_stats["dice"],
            "val_loss": val_stats["loss"],
            "val_wbce": val_stats["wbce"],
            "val_dice": val_stats["dice"],
        }
        history.append(row)

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                history,
                cfg,
                (w_pos, w_neg),
                best_val_loss,
            )
            marker = " *"
        else:
            marker = ""

        print(
            f"Epoch {epoch:03d}/{cfg.epochs:03d} "
            f"lr={row['lr']:.2e} "
            f"train={row['train_loss']:.5f} "
            f"val={row['val_loss']:.5f}{marker}"
        )

        scheduler.step()

    with open(Path(cfg.out_dir) / f"{checkpoint_name}.history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(Path(cfg.out_dir) / f"{checkpoint_name}.config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    return best_path


@torch.no_grad()
def predict_one_patch(
    checkpoint: str | Path,
    patch: str | Path,
    out_dir: str | Path,
    cfg: TrainConfig,
) -> Path:
    model = build_model(cfg)
    load_model_weights(model, checkpoint, cfg.device, strict=True)
    model.eval()

    patch_path = Path(patch)
    if patch_path.suffix == ".npy":
        image = np.load(patch_path)
    elif patch_path.suffix == ".dat":
        image = read_dat_patch(
            patch_path,
            dtype=np.float32,
            patch_size=cfg.patch_size,
        )
    else:
        raise ValueError(f"Unsupported patch file: {patch_path}")

    image = np.squeeze(image).astype(np.float32)
    image = minmax_then_zscore(image)

    x = torch.from_numpy(image[None, None]).float().to(cfg.device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy().astype(np.float32)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{patch_path.stem}_probability.npy"
    np.save(out_path, prob)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train UASS-Net for thrust-fault probability prediction.")

    parser.add_argument("--mode", choices=["synthetic", "transfer", "predict", "sanity"], default="sanity")

    parser.add_argument(
        "--data-root",
        type=str,
        default="outputs/datasets/synthetic",
        help=(
            "Dataset root (default: outputs/datasets/synthetic). "
            "Use outputs/datasets/real_labels for transfer learning."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/training",
        help="Checkpoint, history, configuration, and prediction output directory.",
    )
    parser.add_argument(
        "--file-ext",
        type=str,
        choices=[".npy", ".dat"],
        default=".npy",
        help=(
            "Patch format. Raw .dat datasets must contain headerless C-order "
            "(x, z) square arrays: float32 seismic and uint8 labels; "
            "--patch-size must match the writer."
        ),
    )

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patch-size", type=int, default=256)

    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Pretrained checkpoint; required when --mode transfer is selected.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help=(
            "Freeze the pretrained encoder during transfer learning; valid only "
            "with --mode transfer and --pretrained."
        ),
    )
    parser.add_argument(
        "--no-random-flip",
        action="store_true",
        help="Disable random reversal of physical x (axis 0 in (x, z) arrays).",
    )

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--patch", type=str, default=None)

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)

    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    if args.mode == "synthetic":
        epochs = 66 if args.epochs is None else args.epochs
        lr = 1e-3 if args.lr is None else args.lr
    elif args.mode == "transfer":
        epochs = 50 if args.epochs is None else args.epochs
        lr = 1e-5 if args.lr is None else args.lr
    else:
        epochs = 1 if args.epochs is None else args.epochs
        lr = 1e-3 if args.lr is None else args.lr

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    return TrainConfig(
        data_root=args.data_root,
        out_dir=args.out_dir,
        file_ext=args.file_ext,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        epochs=epochs,
        lr=lr,
        step_size=args.step_size,
        gamma=args.gamma,
        weight_decay=args.weight_decay,
        base_channels=args.base_channels,
        dropout=args.dropout,
        random_flip=not args.no_random_flip,
        seed=args.seed,
        num_workers=args.num_workers,
        device=device,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "transfer" and not args.pretrained:
        parser.error("--pretrained is required when --mode transfer.")
    if args.freeze_encoder and args.mode != "transfer":
        parser.error(
            "--freeze-encoder is only valid with --mode transfer and --pretrained."
        )

    cfg = config_from_args(args)

    if args.mode == "sanity":
        model = build_model(cfg)
        x = torch.randn(2, 1, cfg.patch_size, cfg.patch_size, device=cfg.device)
        y = model(x)
        print(f"Input shape:  {tuple(x.shape)}")
        print(f"Output shape: {tuple(y.shape)}")
        print(f"Parameters:   {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        return

    if args.mode == "synthetic":
        best = train(cfg, "uassnet_synthetic_best.pth")
        print(f"Best synthetic checkpoint: {best}")
        return

    if args.mode == "transfer":
        best = train(
            cfg,
            "uassnet_transfer_best.pth",
            pretrained=args.pretrained,
            freeze=args.freeze_encoder,
        )
        print(f"Best transfer checkpoint: {best}")
        return

    if args.mode == "predict":
        if args.checkpoint is None or args.patch is None:
            raise ValueError("--checkpoint and --patch are required for predict mode.")
        out = predict_one_patch(args.checkpoint, args.patch, cfg.out_dir, cfg)
        print(f"Saved probability map: {out}")
        return


if __name__ == "__main__":
    main()
