from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


def generate_shap_reports(model, X: pd.DataFrame, report_dir: str | Path) -> dict:
    """Generate SHAP artifacts when SHAP supports the current model/environment."""
    path = Path(report_dir)
    path.mkdir(parents=True, exist_ok=True)
    result = {"status": "skipped", "reason": ""}
    try:
        import matplotlib.pyplot as plt
        import shap

        sample = X.tail(min(500, len(X)))
        explainer = shap.Explainer(model, sample)
        values = explainer(sample)
        joblib.dump({"sample": sample, "values": values}, path / "shap_values.pkl")

        shap.plots.beeswarm(values, show=False)
        plt.tight_layout()
        plt.savefig(path / "shap_summary.png")
        plt.close()

        shap.plots.bar(values, show=False)
        plt.tight_layout()
        plt.savefig(path / "shap_feature_impact.png")
        plt.close()

        (path / "shap_report.html").write_text("<html><body><h1>SHAP Report</h1><p>See shap_summary.png and shap_feature_impact.png.</p></body></html>", encoding="utf-8")
        result["status"] = "generated"
    except Exception as exc:
        result["reason"] = str(exc)
        (path / "shap_report.html").write_text(f"<html><body><h1>SHAP unavailable</h1><pre>{exc}</pre></body></html>", encoding="utf-8")
    return result


def explain_trade(model, row: pd.DataFrame, feature_columns: list[str]) -> dict:
    probabilities = model.predict_proba(row[feature_columns])[0]
    return {
        "buy_probability": float(probabilities[1] * 100) if len(probabilities) > 1 else 0.0,
        "sell_probability": 0.0,
        "hold_probability": float((1 - probabilities[1]) * 100) if len(probabilities) > 1 else 100.0,
        "reason": "Primary profitability model ranks this setup by learned feature contribution. Use SHAP report for feature-level detail.",
    }
