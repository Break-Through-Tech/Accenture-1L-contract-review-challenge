# Milestone 2 task 3

Open `notebooks/milestone2_task3_evaluation.ipynb` in Colab. Its functions are embedded, so no repository imports are required. NumPy and pandas are required; scikit-learn is used only by the tests.

1. Load a real `task4_inputs.json` using section A, or copy the definitions and section B into the active task-2 runtime.
2. For live exports, capture each model's row IDs and label names when predicting, verify dataset construction order, and fill the run description. Select only available trained models. Keep initial and weighted BERT arrays separate. Baseline uses `y_val_pred`/`y_val_proba`; initial BERT uses `val_predictions`/`val_probabilities`. Weighted predictions must come from completed weighted training, not its initialization sanity check.
3. Run the final cell. It writes `per_category_metrics.csv`, `model_comparison.csv`, `provenance.json`, and `task4_inputs.json` in `task3_validation_results` relative to the runtime. Task 4 can read the latter with `MODE="real"`.

No real prediction arrays or trained checkpoint are committed here, so no new project scores are included. The saved zero BERT scores are historical and weighted-model scores are unavailable. No retraining, threshold search, push, or merge is performed by task 3.

The bundle uses task 4 schema version 1. Optional model-level `labels` and `row_ids` detect reordered axes; older task-4 exports lack these, so verify their export provenance. All models in a bundle share one truth matrix and chunk list. Cross-bundle comparisons are deliberately not performed. Shape checks cannot prove semantic alignment; the runtime exporter also checks chunk annotations against `mlb.transform` and requires explicit alignment confirmation.

Precision, recall and F1 use chunk–category counts. Undefined denominators produce NaN/CSV blanks. Macro-zero includes all categories with undefined values replaced by zero; macro-defined excludes undefined categories and includes category counts. Micro pools all pairs. No-positive output is flagged. Predictions are retained; a declared threshold requires matching probabilities and `>=` rule. Missing thresholds remain explicitly unknown.

Run checks from the repository root:

```sh
python -m unittest discover -s tests -v
```

Synthetic inputs exist only in tests and never represent project findings. Preserve the JSON bundle with outputs to reproduce results and feed false-negative review. The notebook defaults to no loaded input and does not silently generate a demo.
