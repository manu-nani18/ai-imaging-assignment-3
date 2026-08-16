from __future__ import annotations

import numpy as np


def dice_score(prediction, target, epsilon: float = 1e-7) -> float:
    pred = np.asarray(prediction, dtype=bool)
    true = np.asarray(target, dtype=bool)
    return float((2 * np.logical_and(pred, true).sum() + epsilon) /
                 (pred.sum() + true.sum() + epsilon))


def iou_score(prediction, target, epsilon: float = 1e-7) -> float:
    pred = np.asarray(prediction, dtype=bool)
    true = np.asarray(target, dtype=bool)
    return float((np.logical_and(pred, true).sum() + epsilon) /
                 (np.logical_or(pred, true).sum() + epsilon))

