from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


REGISTRY_FILE = Path("models/registry.json")


def register_model(model_path: str | Path, metrics: dict | None = None, stage: str = "staging") -> dict:
    source = Path(model_path)
    if not source.exists():
        raise FileNotFoundError(f"Model file not found: {source}")
    version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    version_dir = Path("models/versions") / version
    version_dir.mkdir(parents=True, exist_ok=True)
    target = version_dir / source.name
    shutil.copy2(source, target)

    registry = load_registry()
    entry = {"version": version, "path": str(target), "source": str(source), "stage": stage, "metrics": metrics or {}}
    registry["models"].append(entry)
    if stage == "production":
        registry["production"] = version
    save_registry(registry)
    return entry


def promote_model(version: str, stage: str = "production") -> dict:
    registry = load_registry()
    found = False
    for model in registry["models"]:
        if model["version"] == version:
            model["stage"] = stage
            found = True
            if stage == "production":
                registry["production"] = version
    if not found:
        raise ValueError(f"Model version not found: {version}")
    save_registry(registry)
    return registry


def load_registry() -> dict:
    if not REGISTRY_FILE.exists():
        return {"production": None, "models": []}
    return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))


def save_registry(registry: dict) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2), encoding="utf-8")
