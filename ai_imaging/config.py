from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("work/dataset/nuclei_dataset")
    output_dir: Path = Path("outputs")
    image_size: int = 256
    seed: int = 42
    batch_size: int = 8
    epochs: int = 25
    learning_rate: float = 1e-3
    threshold: float = 0.5
    min_object_size: int = 20
    ollama_url: str = "http://localhost:11434"
    vision_model: str = "llama3.2-vision"
    text_model: str = "llama3.2"

