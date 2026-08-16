import numpy as np

from ai_imaging.classical import numeric_summary, otsu_segment, region_features
from ai_imaging.metrics import dice_score, iou_score
from ai_imaging.ollama import OllamaError, parse_json_object, validate_json_fields


def test_overlap_metrics_are_exact_for_identical_masks():
    mask = np.array([[0, 1], [1, 0]], dtype=bool)
    assert dice_score(mask, mask) == 1.0
    assert iou_score(mask, mask) == 1.0


def test_otsu_and_features_find_bright_objects():
    image = np.zeros((64, 64), dtype=float)
    image[5:15, 5:15] = 0.8
    image[30:45, 30:45] = 1.0
    mask, threshold = otsu_segment(image, min_size=5)
    features = region_features(image, mask)
    summary = numeric_summary(features, mask)
    assert 0 <= threshold < 0.8
    assert summary["n_objects"] == 2
    assert summary["mean_area"] == 162.5


def test_json_parser_rejects_schema_drift():
    try:
        parse_json_object('{"a":1,"extra":2}', {"a"})
    except OllamaError:
        pass
    else:
        raise AssertionError("Expected schema mismatch")


def test_semantic_validator_rejects_wrong_types_and_enums():
    record = {"n_objects": "nine", "quality_flag": "maybe"}
    try:
        validate_json_fields(record, integers=("n_objects",),
                             enums={"quality_flag": {"ok", "review", "uncertain"}})
    except OllamaError:
        pass
    else:
        raise AssertionError("Expected semantic validation failure")
