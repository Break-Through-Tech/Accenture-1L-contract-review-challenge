"""Error analysis for the CUAD multi-label classifier."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Simple text checks to help us pick examples for manual review.
TEXT_PATTERNS = {
    "negation": r"\b(not|no|never|without|unless|except|neither)\b",
    "conditional": r"\b(if|provided that|subject to|in the event|upon|unless)\b",
    "cross_reference": r"\b(section|article|schedule|exhibit|appendix)\s+[\w.()-]+",
}


def binary_matrix(values, name, shape):
    matrix = np.asarray(values)
    if matrix.shape != shape or matrix.dtype.kind not in "biuf":
        raise ValueError(f"{name} needs shape {shape} and numeric values")
    if not np.isfinite(matrix).all() or not np.isin(matrix, [0, 1]).all():
        raise ValueError(f"{name} must contain only zeros and ones")
    return matrix.astype(int)


def validate_bundle(data, allow_synthetic=False):
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Expected a version 1 task4_inputs.json file")
    if type(data.get("synthetic")) is not bool:
        raise ValueError("Input must specify synthetic as true or false")
    if data["synthetic"] and not allow_synthetic:
        raise ValueError("Demo data cannot be used for actual project findings")
    if not str(data.get("run_description", "")).strip():
        raise ValueError("Add the model run description before exporting")

    labels = data.get("labels")
    chunks = data.get("chunks")
    model_data = data.get("models")
    if not isinstance(labels, list) or not labels or any(not isinstance(x, str) or not x.strip() for x in labels) or len(set(labels)) != len(labels):
        raise ValueError("Category labels must be unique, nonempty strings")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("No validation chunks found")
    if not isinstance(model_data, dict) or not model_data:
        raise ValueError("No model predictions found")

    chunk_ids = []
    for row in chunks:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in ("contract_title", "chunk_id", "chunk_text")):
            raise ValueError("Every chunk needs a contract title, ID and text")
        if not row["chunk_text"].strip():
            raise ValueError("A validation chunk has empty text")
        chunk_ids.append((row["contract_title"], row["chunk_id"]))
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Repeated contract/chunk ID found")

    shape = (len(chunks), len(labels))
    actual = binary_matrix(data.get("y_true"), "y_true", shape)
    models = {}
    for name, values in model_data.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(values, dict):
            raise ValueError("Invalid model entry")
        predicted = binary_matrix(values.get("predictions"), f"{name} predictions", shape)
        probabilities = None
        if "probabilities" in values:
            probabilities = np.asarray(values["probabilities"])
            if (probabilities.shape != shape or probabilities.dtype.kind not in "biuf"
                    or not np.isfinite(probabilities).all()
                    or ((probabilities < 0) | (probabilities > 1)).any()):
                raise ValueError(f"Invalid probabilities for {name}")
            probabilities = probabilities.astype(float)
        models[name] = (predicted, probabilities)
    return labels, chunks, actual, models


def safe_text(value):
    # Contract text can start with characters Excel treats as formulas.
    value = str(value)
    if value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def ratio(top, bottom):
    return float(top / bottom) if bottom else None


def analyze(data, sample_per_category=5, allow_synthetic=False):
    if not isinstance(sample_per_category, int) or sample_per_category < 1:
        raise ValueError("Choose at least one review example per category")
    labels, chunks, actual, models = validate_bundle(data, allow_synthetic)
    category_rows = []
    review_rows = []
    pattern_rows = []
    model_rows = []

    for model_name, (predicted, probabilities) in models.items():
        true_positive = ((actual == 1) & (predicted == 1)).sum(axis=0)
        false_positive = ((actual == 0) & (predicted == 1)).sum(axis=0)
        false_negative = ((actual == 1) & (predicted == 0)).sum(axis=0)

        for column, label in enumerate(labels):
            tp = int(true_positive[column])
            fp = int(false_positive[column])
            fn = int(false_negative[column])
            category_rows.append({
                "model": model_name, "category": label, "support": tp + fn,
                "tp": tp, "fp": fp, "fn": fn,
                "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
                "f1": ratio(2 * tp, 2 * tp + fp + fn),
            })

            for kind, mask in (
                ("FN", (actual[:, column] == 1) & (predicted[:, column] == 0)),
                ("FP", (actual[:, column] == 0) & (predicted[:, column] == 1)),
            ):
                indices = list(np.flatnonzero(mask))
                # Show borderline predictions first when scores are available.
                if probabilities is not None:
                    indices.sort(key=lambda i: (abs(float(probabilities[i, column]) - 0.5), int(i)))

                for index in indices[:sample_per_category]:
                    chunk = chunks[int(index)]
                    review_rows.append({
                        "model": model_name, "category": label, "error_type": kind,
                        "row_index": int(index),
                        "contract_title": safe_text(chunk["contract_title"]),
                        "chunk_id": safe_text(chunk["chunk_id"]),
                        "chunk_text": safe_text(chunk["chunk_text"]),
                        "probability": None if probabilities is None else float(probabilities[index, column]),
                        "true_label": int(actual[index, column]),
                        "predicted_label": int(predicted[index, column]),
                    })

                texts = [chunks[int(i)]["chunk_text"] for i in indices]
                checks = {name: sum(bool(re.search(regex, t, re.I)) for t in texts)
                          for name, regex in TEXT_PATTERNS.items()}
                checks["long_chunk_400_words"] = sum(len(t.split()) >= 400 for t in texts)
                for name, count in checks.items():
                    pattern_rows.append({
                        "model": model_name, "category": label, "error_type": kind,
                        "pattern": name, "matching_errors": count,
                        "total_errors": len(indices),
                        "fraction": ratio(count, len(indices)),
                    })

        tp, fp, fn = (int(true_positive.sum()), int(false_positive.sum()), int(false_negative.sum()))
        model_rows.append({
            "model": model_name, "rows": len(chunks), "categories": len(labels),
            "tp": tp, "fp": fp, "fn": fn,
            "micro_precision": ratio(tp, tp + fp),
            "micro_recall": ratio(tp, tp + fn),
            "micro_f1": ratio(2 * tp, 2 * tp + fp + fn),
        })

    return {
        "per_category": pd.DataFrame(category_rows),
        "review_samples": pd.DataFrame(review_rows, columns=[
            "model", "category", "error_type", "row_index", "contract_title", "chunk_id",
            "chunk_text", "probability", "true_label", "predicted_label"]),
        "error_patterns": pd.DataFrame(pattern_rows),
        "model_summary": pd.DataFrame(model_rows),
    }


def write_results(data, output_dir, sample_per_category=5, allow_synthetic=False):
    results = analyze(data, sample_per_category, allow_synthetic)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    details = {
        "schema_version": 1, "synthetic": data["synthetic"],
        "run_description": data["run_description"],
        "models": list(data["models"]), "rows": len(data["chunks"]),
        "note": "Text patterns are clues for manual review, not proven causes.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze missed and incorrectly predicted CUAD labels")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", default="task4_results")
    parser.add_argument("--sample-per-category", type=int, default=5)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    tables = write_results(data, args.output, args.sample_per_category, args.allow_synthetic)
    print(tables["model_summary"].to_string(index=False))
    print("Results saved to:", args.output)


if __name__ == "__main__":
    main()
