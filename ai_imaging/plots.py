from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import load_grayscale, load_mask, paired_paths
from .classical import otsu_segment


def save_eda(data_dir: Path, output_dir: Path, n_samples: int = 6) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = paired_paths(data_dir, "train")[:n_samples]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for axis, (path, _) in zip(axes.flat, pairs):
        axis.imshow(load_grayscale(path), cmap="gray", vmin=0, vmax=1)
        axis.set_title(path.stem)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_samples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    values = np.concatenate([load_grayscale(path).ravel() for path, _ in paired_paths(data_dir, "train")])
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.hist(values, bins=64, color="#315f9b", alpha=0.9)
    axis.set(xlabel="Normalised grayscale intensity", ylabel="Pixel count",
             title="Training-set intensity distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_intensity_histogram.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(history, output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss curves")
    axes[0].legend()
    axes[1].plot(epochs, [r["val_dice"] for r in history], label="Dice")
    axes[1].plot(epochs, [r["val_iou"] for r in history], label="IoU")
    axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0, 1), title="Validation overlap")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_prediction_panels(rows, output_path: Path) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)), squeeze=False)
    for i, (name, image, truth, prediction) in enumerate(rows):
        for axis, data, title in zip(axes[i], (image, truth, prediction),
                                     (f"{name}: input", "Ground truth", "U-Net prediction")):
            axis.imshow(data, cmap="gray")
            axis.set_title(title)
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_otsu_panels(data_dir: Path, output_path: Path, n_samples: int = 3,
                     min_size: int = 20) -> None:
    pairs = paired_paths(data_dir, "val")[:n_samples]
    fig, axes = plt.subplots(len(pairs), 3, figsize=(9, 3 * len(pairs)), squeeze=False)
    for row, (image_path, mask_path) in enumerate(pairs):
        image, truth = load_grayscale(image_path), load_mask(mask_path)
        prediction, threshold = otsu_segment(image, min_size)
        titles = (f"{image_path.stem}: input", "Ground truth",
                  f"Otsu prediction (t={threshold:.3f})")
        for axis, data, title in zip(axes[row], (image, truth, prediction), titles):
            axis.imshow(data, cmap="gray")
            axis.set_title(title)
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_method_comparison(metric_frame, output_path: Path) -> None:
    ordered = metric_frame.sort_values("unet_dice")
    positions = np.arange(len(ordered))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(positions - width / 2, ordered["unet_dice"], width, label="U-Net")
    axes[0].bar(positions + width / 2, ordered["otsu_dice"], width, label="Otsu")
    axes[0].set_xticks(positions, ordered["image_id"], rotation=60, ha="right")
    axes[0].set(ylabel="Dice", ylim=(0, 1), title="Per-image segmentation overlap")
    axes[0].legend()
    delta = ordered["unet_dice"] - ordered["otsu_dice"]
    axes[1].bar(positions, delta, color=np.where(delta >= 0, "#2a9d8f", "#e76f51"))
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(positions, ordered["image_id"], rotation=60, ha="right")
    axes[1].set(ylabel="Dice difference (U-Net − Otsu)", title="Where each method performs better")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_method_examples(rows, output_path: Path) -> None:
    """Show the images with largest and smallest U-Net advantage over Otsu."""
    selected = sorted(rows, key=lambda row: row["unet_dice"] - row["otsu_dice"])
    selected = [selected[0], selected[-1]] if len(selected) > 1 else selected
    fig, axes = plt.subplots(len(selected), 4, figsize=(12, 3 * len(selected)), squeeze=False)
    for i, row in enumerate(selected):
        delta = row["unet_dice"] - row["otsu_dice"]
        titles = (f"{row['image_id']}: input", "Ground truth",
                  f"U-Net (Dice {row['unet_dice']:.3f})",
                  f"Otsu (Dice {row['otsu_dice']:.3f}, Δ {delta:+.3f})")
        for axis, data, title in zip(
                axes[i], (row["image"], row["truth"], row["unet"], row["otsu"]), titles):
            axis.imshow(data, cmap="gray")
            axis.set_title(title)
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
