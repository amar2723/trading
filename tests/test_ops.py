from __future__ import annotations

import os
from pathlib import Path

from ops.healthcheck import check_files
from ops.model_registry import load_registry, promote_model, register_model


def test_model_registry_register_and_promote(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("models").mkdir()
    model = Path("models/xgboost_model.pkl")
    model.write_text("model", encoding="utf-8")

    entry = register_model(model, metrics={"win_rate": 60}, stage="staging")
    assert entry["stage"] == "staging"
    assert Path(entry["path"]).exists()

    registry = promote_model(entry["version"], "production")
    assert registry["production"] == entry["version"]
    assert load_registry()["production"] == entry["version"]


def test_file_healthcheck(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for directory in ["data", "models", "logs", "reports"]:
        Path(directory).mkdir()
    assert check_files()
