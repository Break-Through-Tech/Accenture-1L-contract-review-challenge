import copy
import itertools
import hashlib
import json
import math
from .engine import SIGNALS, default_config, validate_config, score_contracts

ORDER = {'Low': 0, 'Medium': 1, 'High': 2}


def advisor_bucket(rating):
    if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5:
        raise ValueError('Advisor rating must be an integer from 1 to 5.')
    return 'Low' if rating <= 2 else 'Medium' if rating == 3 else 'High'


def metrics(records, config):
    from scipy.stats import spearmanr
    rows, _ = score_contracts(records, config)
    if any(row['review_required'] for row in rows):
        raise ValueError('Calibration/evaluation requires fully assessed records.')
    ratings = [r['advisor_rating'] for r in records]
    truth = [advisor_bucket(r) for r in ratings]
    scores = [r['score'] for r in rows]
    predicted = [r['bucket'] for r in rows]
    rho = None
    if len(set(ratings)) > 1 and len(set(scores)) > 1:
        value = float(spearmanr(ratings, scores).statistic)
        rho = value if math.isfinite(value) else None
    high = sum(t == 'High' for t in truth)
    found_high = sum(t == 'High' and p == 'High' for t,p in zip(truth, predicted))
    return {'records': len(rows), 'contracts': len({r['contract_id'] for r in records}),
            'bucket_agreement': sum(a == b for a,b in zip(truth, predicted))/len(rows),
            'mean_bucket_distance': sum(abs(ORDER[a]-ORDER[b]) for a,b in zip(truth,predicted))/len(rows),
            'spearman': rho, 'high_risk_recall': found_high/high if high else None,
            'high_risk_support': high}


def sensitivity(records, config, delta=5.0):
    base, _ = score_contracts(records, config)
    reports = []
    for shift in [-delta, delta]:
        candidate = copy.deepcopy(config)
        candidate['thresholds']['high'] += shift
        if not candidate['thresholds']['medium'] < candidate['thresholds']['high'] <= 100:
            continue
        changed, _ = score_contracts(records, candidate)
        reports.append({'high_threshold': candidate['thresholds']['high'],
                        'bucket_flip_fraction': sum(a['bucket'] != b['bucket'] for a,b in zip(base,changed))/len(base),
                        **metrics(records,candidate)})
    return reports


def calibrate(records, base=None):
    if len(records) < 3 or len({r['contract_id'] for r in records}) < 2:
        raise ValueError('Need at least three advisor-rated records from two contracts; 20–30 recommended.')
    if any(r['split'] not in ['train','validation'] for r in records):
        raise ValueError('Calibration accepts development splits only, never test/synthetic examples.')
    if len({r.get('advisor_rating') for r in records}) < 2:
        raise ValueError('Calibration requires varying advisor ratings.')
    base = copy.deepcopy(base or default_config())
    validate_config(base)
    metrics(records, base)
    best, best_key, count = None, None, 0
    for weights in itertools.product([1.0, 2.0], repeat=4):
        for medium, high in itertools.product([20.,30.,40.], [50.,60.,70.]):
            candidate = copy.deepcopy(base)
            candidate['weights'] = dict(zip(SIGNALS,weights))
            candidate['thresholds'] = {'medium':medium,'high':high}
            report = metrics(records,candidate)
            key = (report['bucket_agreement'], -report['mean_bucket_distance'],
                   report['spearman'] if report['spearman'] is not None else -2,
                   -sum(abs(w-1) for w in weights), -abs(medium-25)-abs(high-60))
            count += 1
            if best_key is None or key > best_key:
                best, best_key = candidate, key
    best['version'] = base['version'] + '-calibrated'
    best['calibration_status'] = 'fitted_to_advisor_development_ratings_not_holdout_validated'
    best['calibration_contract_ids'] = sorted({r['contract_id'] for r in records})
    best['calibration_input_sha256'] = hashlib.sha256(json.dumps(records, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return best, {'status':best['calibration_status'], 'candidates_evaluated':count,
                  'rating_buckets':'1–2 Low; 3 Medium; 4–5 High (proposed, advisor approval pending)',
                  'before_in_sample': metrics(records,base), 'after_in_sample':metrics(records,best),
                  'high_boundary_sensitivity_in_sample':sensitivity(records,best),
                  'limitation':'Development fit only. Freeze config and evaluate different contracts; not risk probabilities.'}


def evaluate_holdout(records, config):
    if config.get('calibration_status') != 'fitted_to_advisor_development_ratings_not_holdout_validated':
        raise ValueError('Use a fitted calibration artifact for holdout evaluation.')
    if not records or any(r['split'] != 'test' for r in records):
        raise ValueError('Held-out evaluation requires test records.')
    overlap = set(config['calibration_contract_ids']) & {r['contract_id'] for r in records}
    if overlap:
        raise ValueError('Calibration and holdout contract IDs overlap.')
    return {'held_out_metrics':metrics(records, config),
            'high_boundary_sensitivity':sensitivity(records,config),
            'config_version':config['version']}
