# Auditable nuclei image-analysis pipeline

This repository implements the coding work for Tasks 1–4: grayscale preprocessing and EDA,
direct VLM descriptions, Otsu/regionprops measurements, a small U-Net, and an unseen-test hybrid
pipeline. Generated numbers are saved to files so they can be checked rather than copied from
free text.

## Dataset

Download the public assignment dataset after cloning the repository:

```powershell
python scripts/setup_data.py
```

This creates `work/dataset/nuclei_dataset`, which is intentionally excluded from Git. The dataset
contains 80 train, 20 validation, and 12 test images. The provided split is preserved. Images are
converted to grayscale and resized to 256×256 when loaded; masks use nearest-neighbour interpolation.

## Setup (PowerShell)

Python 3.10 or newer is recommended. Create a clean environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install [Ollama](https://ollama.com/), start it, and download both local models:

```powershell
ollama pull llama3.2-vision
ollama pull llama3.2
```

Check the non-LLM components:

```powershell
python -m pytest
```

## Running the tasks

Run individual stages in order:

```powershell
python run_pipeline.py eda-vlm
python run_pipeline.py classical-llm
python run_pipeline.py train --epochs 25 --loss bce_dice
python run_pipeline.py hybrid
```

For a hybrid smoke test without new Ollama calls, add `--no-llm`. Existing narratives are retained.
`all` runs every stage, but individual commands make failures easier to diagnose.

Training is reproducibly seeded. It uses CUDA when available and CPU otherwise. The best model is
selected by validation Dice; the test split is not used for model selection.

## Recorded results

The included checkpoint was selected at epoch 24 of a 25-epoch BCE+Dice run. It achieved mean
validation Dice 0.9952 and IoU 0.9904. On the 12-image test split, mean Dice/IoU were 0.9954/0.9908
for U-Net and 0.9784/0.9577 for Otsu. Exact per-image values are retained in `outputs/`.

## Auditable outputs

- `outputs/figures/eda_samples.png` and `eda_intensity_histogram.png`
- `outputs/figures/otsu_validation_panels.png`
- `outputs/task1_vlm_outputs.json` (created after a successful Task 1 VLM run)
- `outputs/task2_region_features.csv` and `task2_numbers_first.json`
- `outputs/unet_best.pt`, `training_history.csv`, validation metrics, curves, and panels
- `outputs/test_records/` (per-image features, JSON, and narrative)
- `outputs/test_aggregated.csv` (required aggregate)
- `outputs/test_unet_vs_otsu.csv` (paired segmentation comparison)
- `outputs/test_metrics_summary.csv` (report-ready mean Dice/IoU table)
- `outputs/figures/unet_vs_otsu.png` (per-image comparison and Dice differences)
- `outputs/figures/method_example_panels.png` (largest/smallest U-Net advantage)
- `outputs/prompts_used.json` (exact prompts for reproducibility)

The small `outputs/` evidence bundle is intentionally versioned. `.venv/`, `work/`, Python caches,
and editor files are excluded by `.gitignore` and should not be committed.

The final hybrid record is requested from the LLM, then `image_id`, `n_objects`, and `mean_area`
are restored programmatically from measured values before it becomes the source of truth. The LLM
classifies density/quality and writes the narrative, but cannot silently change measured fields.
All model prose remains non-clinical and requires human review.

## Design notes and limitations

- A fixed 0.5 U-Net threshold is used for clarity. Threshold tuning must use validation data only.
- Otsu cleanup removes objects and holes smaller than 20 pixels; document any changed value.
- BatchNorm and augmentation improve a small training set, but results can still vary by hardware.
- The dataset is synthetic and cannot establish clinical generalisation.
- The density classes are explicit heuristic thresholds, not clinical categories.
- LLM output is schema-validated. Invalid JSON stops the run instead of being silently accepted.

## Current llama3.2-vision compatibility note

On 16 August 2026, Ollama 0.32.13 for Windows downloaded `llama3.2-vision:latest` but its runner
failed with `unknown model architecture: 'mllama'`. This matches upstream Ollama issue #16547
(a duplicate of #16490). The included Task 1 evidence was therefore generated with an isolated
Ollama 0.5.4 server and the official historical split model/projector, exposed locally as
`llama3.2-vision:11b` on port 11555:

```powershell
python run_pipeline.py eda-vlm `
  --ollama-url http://127.0.0.1:11555 `
  --vision-model llama3.2-vision:11b
```

The legacy Ollama binaries and model blobs were kept outside this repository. When the upstream
Windows compatibility issue is resolved, the standard `llama3.2-vision` setup above can be used.
`outputs/task1_vlm_outputs.json` contains the completed naive comparison and three validated,
non-identical structured runs.

No supplied validation or test image has higher whole-image Dice with Otsu than with the trained
U-Net. Otsu is nevertheless competitive on simple, high-contrast images such as `val_011`.
