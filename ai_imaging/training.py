from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import NucleiDataset
from .model import UNet, segmentation_loss


def device_for_training() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_metrics(logits, targets, threshold: float = 0.5) -> tuple[float, float]:
    pred = torch.sigmoid(logits) >= threshold
    true = targets >= 0.5
    dims = (1, 2, 3)
    intersection = (pred & true).sum(dims).float()
    pred_sum = pred.sum(dims).float()
    true_sum = true.sum(dims).float()
    union = (pred | true).sum(dims).float()
    dice = ((2 * intersection + 1e-7) / (pred_sum + true_sum + 1e-7)).mean()
    iou = ((intersection + 1e-7) / (union + 1e-7)).mean()
    return float(dice), float(iou)


@torch.no_grad()
def evaluate(model, loader, device, loss_mode: str, threshold: float = 0.5):
    model.eval()
    loss_sum = dice_sum = iou_sum = 0.0
    n_images = 0
    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        batch_size = len(images)
        loss_sum += float(segmentation_loss(logits, masks, loss_mode)) * batch_size
        dice, iou = batch_metrics(logits, masks, threshold)
        dice_sum += dice * batch_size
        iou_sum += iou * batch_size
        n_images += batch_size
    if n_images == 0:
        raise ValueError("Cannot evaluate an empty data loader")
    return {"loss": loss_sum / n_images, "dice": dice_sum / n_images,
            "iou": iou_sum / n_images}


def train_unet(data_dir: Path, epochs: int, batch_size: int, learning_rate: float,
               loss_mode: str = "bce_dice", seed: int = 42, threshold: float = 0.5):
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(NucleiDataset(data_dir, "train", augment=True), batch_size=batch_size,
                              shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(NucleiDataset(data_dir, "val"), batch_size=batch_size,
                            shuffle=False, num_workers=0)
    device = device_for_training()
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history, best_dice, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(images), masks, loss_mode)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        val = evaluate(model, val_loader, device, loss_mode, threshold)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)),
               "val_loss": val["loss"], "val_dice": val["dice"], "val_iou": val["iou"]}
        history.append(row)
        print(f"epoch={epoch:03d} train={row['train_loss']:.4f} val={val['loss']:.4f} "
              f"dice={val['dice']:.4f} iou={val['iou']:.4f}")
        if val["dice"] > best_dice:
            best_dice, best_state = val["dice"], copy.deepcopy(model.state_dict())
    if best_state is None:
        raise ValueError("Training requires at least one epoch")
    model.load_state_dict(best_state)
    return model, history, device


@torch.no_grad()
def predict_mask(model, image_tensor, device, threshold: float = 0.5):
    model.eval()
    logits = model(image_tensor.unsqueeze(0).to(device))
    probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return probability, probability >= threshold
