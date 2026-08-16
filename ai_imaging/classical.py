from __future__ import annotations

import json

import numpy as np
import pandas as pd
from skimage import filters, measure, morphology


FEATURE_COLUMNS = (
    "label", "area", "eccentricity", "solidity", "mean_intensity",
    "perimeter", "major_axis_length", "minor_axis_length",
)


def otsu_segment(image: np.ndarray, min_size: int = 20) -> tuple[np.ndarray, float]:
    threshold = float(filters.threshold_otsu(image))
    mask = image > threshold
    # max_size is the forward-compatible scikit-image 0.26+ name. Objects at
    # exactly the cutoff are retained by using min_size - 1.
    mask = morphology.remove_small_objects(mask, max_size=min_size - 1)
    mask = morphology.remove_small_holes(mask, max_size=min_size - 1)
    return mask.astype(bool), threshold


def region_features(image: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    labels = measure.label(mask)
    props = measure.regionprops_table(
        labels,
        intensity_image=image,
        properties=FEATURE_COLUMNS,
    )
    return pd.DataFrame(props)


def density_class(n_objects: int, foreground_fraction: float) -> str:
    if n_objects < 15 and foreground_fraction < 0.12:
        return "sparse"
    if n_objects >= 55 or foreground_fraction >= 0.35:
        return "dense"
    return "normal"


def numeric_summary(features: pd.DataFrame, mask: np.ndarray) -> dict:
    n = int(len(features))
    fraction = float(np.mean(mask))
    def mean_or_zero(column: str) -> float:
        return float(features[column].mean()) if n else 0.0
    return {
        "n_objects": n,
        "foreground_fraction": round(fraction, 4),
        "mean_area": round(mean_or_zero("area"), 2),
        "mean_eccentricity": round(mean_or_zero("eccentricity"), 4),
        "mean_solidity": round(mean_or_zero("solidity"), 4),
        "mean_intensity": round(mean_or_zero("mean_intensity"), 4),
        "density_class_rule": density_class(n, fraction),
    }


def summary_as_numbers_only(summary: dict) -> str:
    allowed = {k: v for k, v in summary.items() if k != "density_class_rule"}
    return json.dumps(allowed, separators=(",", ":"))
