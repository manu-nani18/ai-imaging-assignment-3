from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = (
    "https://raw.githubusercontent.com/Nickolay-K/"
    "Assingnment-3-dataset/main/nuclei_dataset.zip"
)


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe archive path: {member.filename}")
        bundle.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the assignment dataset")
    parser.add_argument("--destination", type=Path, default=Path("work/dataset"))
    args = parser.parse_args()
    expected = args.destination / "nuclei_dataset"
    if expected.exists():
        print(f"Dataset already exists: {expected}")
        return
    args.destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temporary:
        archive = Path(temporary.name)
    try:
        print(f"Downloading {DATASET_URL}")
        with urllib.request.urlopen(DATASET_URL, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        safe_extract(archive, args.destination)
    finally:
        archive.unlink(missing_ok=True)
    if not expected.exists():
        raise FileNotFoundError(f"Archive did not contain {expected.name}")
    print(f"Dataset ready: {expected}")


if __name__ == "__main__":
    main()
