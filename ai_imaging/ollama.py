from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import requests


class OllamaError(RuntimeError):
    pass


def _post(url: str, payload: dict, timeout: int = 900) -> dict:
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if not response.ok:
            raise OllamaError(
                f"Ollama returned HTTP {response.status_code} at {url}: {response.text[:1000]}"
            )
        return response.json()
    except OllamaError:
        raise
    except (requests.RequestException, ValueError) as exc:
        raise OllamaError(f"Ollama request failed at {url}: {exc}") from exc


def parse_json_object(text: str, required_keys: set[str]) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OllamaError(f"Model output was not valid JSON: {text!r}") from exc
    if not isinstance(value, dict):
        raise OllamaError("Model output must be one JSON object")
    missing = required_keys - value.keys()
    extra = value.keys() - required_keys
    if missing or extra:
        raise OllamaError(f"Schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    return value


def validate_json_fields(record: dict[str, Any], *, strings: tuple[str, ...] = (),
                         integers: tuple[str, ...] = (), numbers: tuple[str, ...] = (),
                         string_lists: tuple[str, ...] = (),
                         enums: dict[str, set[str]] | None = None) -> None:
    invalid = []
    invalid.extend(key for key in strings if not isinstance(record[key], str))
    invalid.extend(key for key in integers
                   if not isinstance(record[key], int) or isinstance(record[key], bool))
    invalid.extend(key for key in numbers
                   if not isinstance(record[key], (int, float)) or isinstance(record[key], bool))
    invalid.extend(key for key in string_lists
                   if not isinstance(record[key], list)
                   or not all(isinstance(item, str) for item in record[key]))
    for key, allowed in (enums or {}).items():
        if record[key] not in allowed:
            invalid.append(key)
    if invalid:
        raise OllamaError(f"Invalid field types or values: {sorted(set(invalid))}")


def generate_text(base_url: str, model: str, prompt: str, temperature: float = 0.1,
                  json_mode: bool = True) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": temperature}}
    if json_mode:
        payload["format"] = "json"
    result = _post(f"{base_url}/api/generate", payload)
    return str(result["response"]).strip()


def describe_image(base_url: str, model: str, prompt: str, image_path: Path,
                   temperature: float = 0.2, json_mode: bool = True) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": model, "prompt": prompt, "images": [encoded], "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"
    result = _post(f"{base_url}/api/generate", payload)
    return str(result["response"]).strip()
