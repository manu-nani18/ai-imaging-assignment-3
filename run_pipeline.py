"""Command-line entry point for the four assignment pipeline tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ai_imaging.classical import numeric_summary, otsu_segment, region_features, summary_as_numbers_only
from ai_imaging.config import Config
from ai_imaging.data import NucleiDataset, image_paths, load_grayscale, seed_everything
from ai_imaging.metrics import dice_score, iou_score
from ai_imaging.model import UNet
from ai_imaging.ollama import (OllamaError, describe_image, generate_text,
                               parse_json_object, validate_json_fields)
from ai_imaging.plots import (save_eda, save_method_comparison, save_method_examples, save_otsu_panels,
                              save_prediction_panels, save_training_curves)
from ai_imaging.prompts import HYBRID_RECORD_PROMPT, NAIVE_VISION_PROMPT, NARRATIVE_PROMPT, NUMBERS_PROMPT, OPTIMISED_VISION_PROMPT
from ai_imaging.training import predict_mask, train_unet


VISION_KEYS = {"modality", "tissue_type", "notable_features", "image_quality"}
NUMERIC_KEYS = {"n_objects", "density_class", "shape_regularity", "quality_flag"}
HYBRID_KEYS = {"image_id", "n_objects", "mean_area", "density_class", "quality_flag"}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def save_prompts(output_dir: Path) -> None:
    write_json(output_dir / "prompts_used.json", {
        "naive_vision": NAIVE_VISION_PROMPT,
        "optimised_vision": OPTIMISED_VISION_PROMPT,
        "numbers_first_template": NUMBERS_PROMPT.format(
            measurements="<MEASUREMENTS_JSON>"
        ),
        "hybrid_record_template": HYBRID_RECORD_PROMPT.format(
            measurements="<MEASUREMENTS_JSON>"
        ),
        "narrative_template": NARRATIVE_PROMPT.format(record="<STRUCTURED_JSON>"),
    })


def task1(config: Config, repetitions: int = 3) -> None:
    figure_dir = config.output_dir / "figures"
    save_eda(config.data_dir, figure_dir)
    save_otsu_panels(config.data_dir, figure_dir / "otsu_validation_panels.png",
                     min_size=config.min_object_size)
    source_image = image_paths(config.data_dir, "train")[0]
    representative = config.output_dir / "representative_grayscale_256.png"
    grayscale = (load_grayscale(source_image, config.image_size) * 255).round().astype(np.uint8)
    Image.fromarray(grayscale).save(representative)
    records = {"image": representative.name, "naive": None, "optimised_runs": []}
    records["naive"] = describe_image(config.ollama_url, config.vision_model, NAIVE_VISION_PROMPT,
                                      representative, temperature=0.4, json_mode=False)
    for _ in range(repetitions):
        raw = describe_image(config.ollama_url, config.vision_model, OPTIMISED_VISION_PROMPT,
                             representative, temperature=0.3, json_mode=True)
        record = parse_json_object(raw, VISION_KEYS)
        validate_json_fields(
            record,
            strings=("modality", "tissue_type", "image_quality"),
            string_lists=("notable_features",),
            enums={"image_quality": {"good", "acceptable", "poor", "uncertain"}},
        )
        records["optimised_runs"].append(record)
    write_json(config.output_dir / "task1_vlm_outputs.json", records)
    print("Task 1 outputs saved.")


def task2(config: Config) -> None:
    representative = image_paths(config.data_dir, "train")[0]
    image = load_grayscale(representative, config.image_size)
    mask, threshold = otsu_segment(image, config.min_object_size)
    features = region_features(image, mask)
    summary = numeric_summary(features, mask)
    summary["otsu_threshold"] = round(threshold, 4)
    measurements = summary_as_numbers_only(summary)
    raw = generate_text(config.ollama_url, config.text_model,
                        NUMBERS_PROMPT.format(measurements=measurements))
    record = parse_json_object(raw, NUMERIC_KEYS)
    validate_json_fields(
        record,
        integers=("n_objects",),
        enums={
            "density_class": {"sparse", "normal", "dense", "uncertain"},
            "shape_regularity": {"regular", "mixed", "irregular", "uncertain"},
            "quality_flag": {"ok", "review", "uncertain"},
        },
    )
    record["n_objects"] = summary["n_objects"]
    features.to_csv(config.output_dir / "task2_region_features.csv", index=False)
    write_json(config.output_dir / "task2_numbers_first.json",
               {"image": representative.name, "measurements": summary, "llm_record": record})
    print("Task 2 outputs saved.")


def task3(config: Config, loss_mode: str) -> None:
    model, history, device = train_unet(config.data_dir, config.epochs, config.batch_size,
                                        config.learning_rate, loss_mode, config.seed, config.threshold)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {"model_state": model.state_dict(), "loss_mode": loss_mode,
                  "threshold": config.threshold, "image_size": config.image_size}
    torch.save(checkpoint, config.output_dir / "unet_best.pt")
    pd.DataFrame(history).to_csv(config.output_dir / "training_history.csv", index=False)
    save_training_curves(history, config.output_dir / "figures" / "training_curves.png")

    dataset = NucleiDataset(config.data_dir, "val", config.image_size)
    rows, metrics = [], []
    for index in range(len(dataset)):
        image_tensor, mask_tensor, name = dataset[index]
        _, prediction = predict_mask(model, image_tensor, device, config.threshold)
        truth = mask_tensor[0].numpy().astype(bool)
        metrics.append({"image_id": name, "dice": dice_score(prediction, truth),
                        "iou": iou_score(prediction, truth)})
        if len(rows) < 3:
            rows.append((name, image_tensor[0].numpy(), truth, prediction))
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(config.output_dir / "validation_metrics_per_image.csv", index=False)
    write_json(config.output_dir / "validation_metrics_summary.json", {
        "loss_mode": loss_mode, "mean_dice": float(metric_frame.dice.mean()),
        "mean_iou": float(metric_frame.iou.mean()), "n_images": len(metric_frame),
    })
    save_prediction_panels(rows, config.output_dir / "figures" / "validation_panels.png")
    print("Task 3 outputs saved.")


def load_checkpoint(config: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config.output_dir / "unet_best.pt", map_location=device, weights_only=True)
    if checkpoint["image_size"] != config.image_size:
        raise ValueError("Checkpoint image size does not match the configured image size")
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, device


def task4(config: Config, use_llm: bool = True) -> None:
    model, device = load_checkpoint(config)
    dataset = NucleiDataset(config.data_dir, "test", config.image_size)
    records, metric_rows, example_rows = [], [], []
    record_dir = config.output_dir / "test_records"
    record_dir.mkdir(parents=True, exist_ok=True)
    for index in range(len(dataset)):
        image_tensor, truth_tensor, name = dataset[index]
        _, prediction = predict_mask(model, image_tensor, device, config.threshold)
        image = image_tensor[0].numpy()
        features = region_features(image, prediction)
        summary = numeric_summary(features, prediction)
        measured_record = {
            "image_id": name,
            "n_objects": summary["n_objects"],
            "mean_area": summary["mean_area"],
            "density_class": summary["density_class_rule"],
            "quality_flag": "review" if summary["n_objects"] == 0 else "ok",
        }
        if use_llm:
            raw_record = generate_text(
                config.ollama_url, config.text_model,
                HYBRID_RECORD_PROMPT.format(measurements=json.dumps(measured_record)), temperature=0.1,
            )
            record = parse_json_object(raw_record, HYBRID_KEYS)
            validate_json_fields(
                record,
                strings=("image_id",),
                integers=("n_objects",),
                numbers=("mean_area",),
                enums={
                    "density_class": {"sparse", "normal", "dense", "uncertain"},
                    "quality_flag": {"ok", "review", "uncertain"},
                },
            )
            # Preserve measured provenance even if the LLM changes copied values.
            for key in ("image_id", "n_objects", "mean_area"):
                record[key] = measured_record[key]
            narrative = generate_text(config.ollama_url, config.text_model,
                                      NARRATIVE_PROMPT.format(record=json.dumps(record)),
                                      temperature=0.1, json_mode=False)
        else:
            record = measured_record
            existing_path = record_dir / f"{name}.json"
            if existing_path.exists():
                narrative = json.loads(existing_path.read_text(encoding="utf-8")).get(
                    "narrative", "LLM narrative not requested."
                )
            else:
                narrative = "LLM narrative not requested."
        payload = {"record": record, "narrative": narrative}
        write_json(record_dir / f"{name}.json", payload)
        features.to_csv(record_dir / f"{name}_features.csv", index=False)
        records.append(record)
        truth = truth_tensor[0].numpy().astype(bool)
        otsu, _ = otsu_segment(image, config.min_object_size)
        metric_rows.append({
            "image_id": name,
            "unet_dice": dice_score(prediction, truth), "unet_iou": iou_score(prediction, truth),
            "otsu_dice": dice_score(otsu, truth), "otsu_iou": iou_score(otsu, truth),
        })
        example_rows.append({**metric_rows[-1], "image": image, "truth": truth,
                             "unet": prediction, "otsu": otsu})
    pd.DataFrame(records).to_csv(config.output_dir / "test_aggregated.csv", index=False)
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(config.output_dir / "test_unet_vs_otsu.csv", index=False)
    pd.DataFrame([
        {"method": "U-Net", "mean_dice": metric_frame.unet_dice.mean(),
         "mean_iou": metric_frame.unet_iou.mean(), "n_images": len(metric_frame)},
        {"method": "Otsu", "mean_dice": metric_frame.otsu_dice.mean(),
         "mean_iou": metric_frame.otsu_iou.mean(), "n_images": len(metric_frame)},
    ]).to_csv(config.output_dir / "test_metrics_summary.csv", index=False)
    save_method_comparison(metric_frame, config.output_dir / "figures" / "unet_vs_otsu.png")
    save_method_examples(example_rows, config.output_dir / "figures" / "method_example_panels.png")
    print("Task 4 outputs saved.")


def build_config(args) -> Config:
    return Config(data_dir=Path(args.data_dir), output_dir=Path(args.output_dir),
                  epochs=getattr(args, "epochs", 25), batch_size=getattr(args, "batch_size", 8),
                  ollama_url=getattr(args, "ollama_url", "http://localhost:11434"),
                  vision_model=getattr(args, "vision_model", "llama3.2-vision"),
                  text_model=getattr(args, "text_model", "llama3.2"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("eda-vlm", "classical-llm", "train", "hybrid", "all"))
    parser.add_argument("--data-dir", default="work/dataset/nuclei_dataset")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--loss", choices=("bce", "dice", "bce_dice"), default="bce_dice")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--vision-model", default="llama3.2-vision")
    parser.add_argument("--text-model", default="llama3.2")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip hybrid LLM calls and retain any existing narratives")
    args = parser.parse_args()
    config = build_config(args)
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_prompts(config.output_dir)
    (config.output_dir / "figures").mkdir(exist_ok=True)
    try:
        if args.command in ("eda-vlm", "all"):
            task1(config)
        if args.command in ("classical-llm", "all"):
            task2(config)
        if args.command in ("train", "all"):
            task3(config, args.loss)
        if args.command in ("hybrid", "all"):
            task4(config, not args.no_llm)
    except OllamaError as exc:
        raise SystemExit(f"Ollama step failed: {exc}") from exc


if __name__ == "__main__":
    main()
