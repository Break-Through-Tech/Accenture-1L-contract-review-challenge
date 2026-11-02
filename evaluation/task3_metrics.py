"""Task 3 evaluation. Bundles are also readable by the task-4 notebook."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def validate_bundle(bundle):
    if bundle.get('schema_version') != 1 or not isinstance(bundle.get('synthetic'), bool):
        raise ValueError('Expected schema_version=1 and an explicit synthetic boolean.')
    if not isinstance(bundle.get('run_description'), str) or not bundle['run_description'].strip():
        raise ValueError('A run description is required.')
    labels = bundle['labels']
    if not labels or any(not isinstance(x, str) or not x.strip() for x in labels) or len(set(labels)) != len(labels):
        raise ValueError('Category names must be nonempty and unique.')
    chunks = pd.DataFrame(bundle['chunks'])
    fields = ['contract_title', 'chunk_id', 'chunk_text']
    if chunks.empty or not set(fields).issubset(chunks) or chunks[fields].isna().any().any():
        raise ValueError('Missing chunk metadata.')
    if chunks.duplicated(['contract_title', 'chunk_id']).any():
        raise ValueError('Duplicate contract/chunk identifiers.')
    truth = np.asarray(bundle['y_true'])
    shape = (len(chunks), len(labels))
    if truth.shape != shape or not np.isin(truth, [0, 1]).all():
        raise ValueError('Invalid truth matrix or dimensions.')
    if not bundle['models']:
        raise ValueError('No models supplied.')
    for name, entry in bundle['models'].items():
        pred = np.asarray(entry['predictions'])
        if pred.shape != shape or not np.isin(pred, [0, 1]).all():
            raise ValueError(f'{name}: invalid predictions.')
        # Optional explicit axes protect against same-shaped permutations.
        if 'labels' in entry and entry['labels'] != labels:
            raise ValueError(f'{name}: label order mismatch.')
        ids = [[r['contract_title'], r['chunk_id']] for r in bundle['chunks']]
        if 'row_ids' in entry and entry['row_ids'] != ids:
            raise ValueError(f'{name}: row order mismatch.')
        if 'probabilities' in entry:
            p = np.asarray(entry['probabilities'], dtype=float)
            if p.shape != shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise ValueError(f'{name}: invalid probabilities.')
            if 'threshold' in entry:
                t = np.asarray(entry['threshold'], dtype=float)
                if t.shape not in [(), (len(labels),)] or not np.isfinite(t).all() or ((t < 0) | (t > 1)).any():
                    raise ValueError(f'{name}: invalid threshold.')
                if entry.get('threshold_rule') != '>=' or not np.array_equal(pred, p >= t):
                    raise ValueError(f'{name}: predictions do not match declared >= threshold.')
        elif 'threshold' in entry:
            raise ValueError(f'{name}: probabilities required to verify threshold.')
    return truth.astype(int), labels


def ratio(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)


def evaluate(bundle):
    truth, labels = validate_bundle(bundle)
    tables, summaries = [], []
    for name, entry in bundle['models'].items():
        pred = np.asarray(entry['predictions'])
        tp = ((truth == 1) & (pred == 1)).sum(0)
        fp = ((truth == 0) & (pred == 1)).sum(0)
        fn = ((truth == 1) & (pred == 0)).sum(0)
        support, positive = truth.sum(0), pred.sum(0)
        p, r, f = ratio(tp, tp+fp), ratio(tp, tp+fn), ratio(2*tp, 2*tp+fp+fn)
        tables.append(pd.DataFrame(dict(model=name, category=labels, precision=p, recall=r, f1=f,
            support=support, predicted_positive=positive, TP=tp, FP=fp, FN=fn,
            precision_defined=positive > 0, recall_defined=support > 0,
            f1_defined=(2*tp+fp+fn) > 0)))
        row = dict(model=name, chunks=len(truth), categories=len(labels), support=int(support.sum()),
                   predicted_positive=int(positive.sum()), no_positive_predictions=bool(positive.sum() == 0),
                   threshold=json.dumps(entry.get('threshold', 'unknown / original predictions retained')))
        for metric, vals, num, den in [('precision', p, tp.sum(), (tp+fp).sum()),
                                      ('recall', r, tp.sum(), (tp+fn).sum()),
                                      ('f1', f, 2*tp.sum(), (2*tp+fp+fn).sum())]:
            row['micro_'+metric] = float(ratio(num, den))
            # Stable comparison across all categories, compatible with sklearn zero_division=0.
            row['macro_'+metric+'_zero'] = float(np.nan_to_num(vals).mean())
            row['macro_'+metric+'_defined'] = float(vals[np.isfinite(vals)].mean()) if np.isfinite(vals).any() else np.nan
            row[metric+'_defined_categories'] = int(np.isfinite(vals).sum())
        summaries.append(row)
    return pd.concat(tables, ignore_index=True), pd.DataFrame(summaries)


def export_results(bundle, output_dir):
    categories, overall = evaluate(bundle)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    categories.to_csv(out / 'per_category_metrics.csv', index=False)
    overall.to_csv(out / 'model_comparison.csv', index=False)
    # Preserve predictions and metadata for task 4, including synthetic provenance.
    (out / 'task4_inputs.json').write_text(json.dumps(bundle, indent=2, allow_nan=False))
    provenance = dict(synthetic=bundle['synthetic'], run_description=bundle['run_description'],
        input_sha256=hashlib.sha256(json.dumps(bundle, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        undefined_metrics='NaN / blank in CSV; macro_*_zero uses zero for undefined categories; macro_*_defined excludes them.',
        comparison='Only models in this single shared validation bundle are compared. No threshold tuning is performed.')
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    return categories, overall


def bundle_from_runtime(chunks, truth, mlb, models, run_description, alignment_confirmed=False):
    """Models must have predictions, labels, row_ids and optional probabilities/threshold."""
    if not alignment_confirmed:
        raise ValueError('Confirm each model used these exact validation rows and category order.')
    chunks = chunks.reset_index(drop=True)
    truth = np.asarray(truth)
    if 'categories' not in chunks or not np.array_equal(mlb.transform(chunks['categories']), truth):
        raise ValueError('Chunk annotations do not match y_val in this order.')
    records = chunks[['contract_title', 'chunk_id', 'chunk_text']].to_dict('records')
    labels = [str(x) for x in mlb.classes_]
    entries = {}
    for name, values in models.items():
        if 'labels' not in values or 'row_ids' not in values:
            raise ValueError(f'{name}: record labels and row_ids at prediction time.')
        entries[name] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in values.items()}
    bundle = dict(schema_version=1, synthetic=False, run_description=run_description,
                  labels=labels, chunks=records, y_true=truth.tolist(), models=entries)
    validate_bundle(bundle)
    return bundle
