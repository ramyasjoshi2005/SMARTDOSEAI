"""Load offline training artifacts for the ML results dashboard."""

from __future__ import annotations

import json
import os

import pandas as pd

from clinical_units import readable_feature_name
from paths import project_path

DATASETS = ["diabetes", "heart", "kidney", "hypertension", "dialysis"]


def _read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_model_performance():
    rows = []
    for name in DATASETS:
        meta = _read_json(project_path("model_results", f"{name}_best_model.json"))
        comparison_path = project_path("model_results", f"{name}_model_comparison.csv")
        roc = None
        f1 = meta.get("f1_score")
        accuracy = meta.get("accuracy")
        if os.path.exists(comparison_path):
            frame = pd.read_csv(comparison_path)
            best_name = meta.get("best_model")
            match = frame
            if best_name and "Model" in frame.columns:
                subset = frame[frame["Model"].astype(str).str.lower() == str(best_name).lower()]
                if len(subset):
                    match = subset
            if len(match):
                row = match.iloc[0]
                accuracy = row.get("Test_Accuracy", accuracy)
                f1 = row.get("Test_Macro_F1", f1)
                roc = row.get("Test_ROC_AUC")
        rows.append(
            {
                "dataset": name.title(),
                "best_model": meta.get("best_model", "—"),
                "accuracy": _pct(accuracy),
                "f1": _pct(f1),
                "roc_auc": _pct(roc) if roc is not None else "—",
            }
        )
    return rows


def load_model_comparisons():
    tables = {}
    for name in DATASETS:
        path = project_path("model_results", f"{name}_model_comparison.csv")
        if os.path.exists(path):
            tables[name] = pd.read_csv(path).to_dict("records")
        else:
            tables[name] = []
    return tables


def load_shap_importance():
    tables = {}
    for name in DATASETS:
        path = project_path("model_results", "shap", name, "global_feature_importance.csv")
        if not os.path.exists(path):
            tables[name] = []
            continue
        frame = pd.read_csv(path).head(10)
        rows = []
        for _, item in frame.iterrows():
            rows.append(
                {
                    "feature": readable_feature_name(item.get("feature")),
                    "mean_abs_shap": round(float(item.get("mean_abs_shap") or 0), 4),
                }
            )
        tables[name] = rows
    return tables


def _pct(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number <= 1:
        number *= 100
    return f"{number:.1f}%"
