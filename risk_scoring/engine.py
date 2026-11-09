import copy
import math

SIGNALS = {
    'missing_protections': ('missing_liability', 'missing_indemnification', 'missing_ip_ownership'),
    'financial_exposure': ('uncapped_liability', 'open_ended_indemnification'),
    'one_sided_rights': ('one_sided_termination',),
    'renewal_ip_clarity': ('renewal_without_notice', 'ambiguous_ip_ownership'),
}
FLAGS = tuple(flag for flags in SIGNALS.values() for flag in flags)
CRITICAL = ('uncapped_liability', 'missing_indemnification')


def default_config():
    return {
        'schema_version': 1,
        'version': 'advisor-proposal-v1',
        'calibration_status': 'provisional_not_advisor_calibrated',
        'weights': {name: 1.0 for name in SIGNALS},
        'severity': {flag: 1.0 for flag in FLAGS},
        'thresholds': {'medium': 25.0, 'high': 60.0},
        'calibration_contract_ids': [],
    }


def validate_config(config):
    if config.get('schema_version') != 1 or not isinstance(config.get('version'), str):
        raise ValueError('Invalid configuration version.')
    if set(config['weights']) != set(SIGNALS) or set(config['severity']) != set(FLAGS):
        raise ValueError('Configuration must define exactly the four signals and eight flags.')
    for field in ['weights', 'severity']:
        for value in config[field].values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Weights/severities must be finite nonnegative numbers.')
    if sum(config['weights'].values()) <= 0 or any(v > 1 for v in config['severity'].values()):
        raise ValueError('Positive total weight and severities in [0,1] required.')
    m, h = config['thresholds']['medium'], config['thresholds']['high']
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in [m, h]) or not 0 < m < h <= 100:
        raise ValueError('Require 0 < medium < high <= 100.')


def validate_record(record):
    for key in ['record_id', 'contract_id', 'category', 'text', 'source', 'split']:
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f'Nonempty {key} required.')
    if record['split'] not in ['train', 'validation', 'test', 'synthetic']:
        raise ValueError('Unknown split.')
    if set(record['assessments']) != set(FLAGS):
        raise ValueError('Every risk flag must be assessed, unknown or not_applicable.')
    for flag, assessment in record['assessments'].items():
        status = assessment.get('status')
        if status not in ['present', 'absent', 'unknown', 'not_applicable']:
            raise ValueError(f'{flag}: invalid assessment status.')
        if status != 'unknown':
            if not isinstance(assessment.get('rationale'), str) or not assessment['rationale'].strip():
                raise ValueError(f'{flag}: assessed flags require rationale.')
            if not isinstance(assessment.get('reviewer'), str) or not assessment['reviewer'].strip():
                raise ValueError(f'{flag}: assessed flags require reviewer/provenance.')
        if status == 'present':
            if flag.startswith('missing_'):
                if assessment.get('scope') != 'full_contract':
                    raise ValueError('Missing protection requires full-contract review, not missing chunk predictions.')
            else:
                evidence = assessment.get('evidence')
                if not isinstance(evidence, str) or not evidence.strip() or evidence not in record['text']:
                    raise ValueError(f'{flag}: evidence must be an exact excerpt of the record text.')
    if any(record['assessments'][missing]['status'] == 'present' and
           record['assessments'][other]['status'] == 'present'
           for missing, other in [('missing_liability', 'uncapped_liability'),
                                  ('missing_indemnification', 'open_ended_indemnification'),
                                  ('missing_ip_ownership', 'ambiguous_ip_ownership')]):
        raise ValueError('Missing protection conflicts with a present clause characteristic.')


def bucket(score, config):
    if score >= config['thresholds']['high']:
        return 'High'
    if score >= config['thresholds']['medium']:
        return 'Medium'
    return 'Low'


def score_record(record, config=None):
    config = default_config() if config is None else config
    validate_config(config)
    validate_record(record)
    signal_values, unknown, reasons = {}, [], []
    for name, flags in SIGNALS.items():
        lower, upper = [], []
        for flag in flags:
            a = record['assessments'][flag]
            status = a['status']
            severity = config['severity'][flag]
            lower.append(severity if status == 'present' else 0.0)
            upper.append(severity if status in ['present', 'unknown'] else 0.0)
            if status == 'unknown':
                unknown.append(flag)
            if status == 'present':
                reasons.append({'flag': flag, **copy.deepcopy(a)})
        signal_values[name] = {'lower': max(lower), 'upper': max(upper)}
    total = sum(config['weights'].values())
    lower = 100 * sum(config['weights'][s] * v['lower'] for s,v in signal_values.items()) / total
    upper = 100 * sum(config['weights'][s] * v['upper'] for s,v in signal_values.items()) / total
    critical = [f for f in CRITICAL if record['assessments'][f]['status'] == 'present']
    missing = any(record['assessments'][f]['status'] == 'present' for f in SIGNALS['missing_protections'])
    if critical:
        lower, upper = max(lower, config['thresholds']['high']), max(upper, config['thresholds']['high'])
    elif missing:
        lower, upper = max(lower, config['thresholds']['medium']), max(upper, config['thresholds']['medium'])
    return {
        'record_id': record['record_id'], 'contract_id': record['contract_id'], 'category': record['category'],
        'split': record['split'], 'source': record['source'], 'text': record['text'],
        'config_version': config['version'], 'calibration_status': config['calibration_status'],
        'score': round(lower, 6) if not unknown else None,
        'score_lower': round(lower, 6), 'score_upper': round(upper, 6),
        'bucket': bucket(lower, config) if not unknown or critical else None,
        'minimum_bucket': bucket(lower, config), 'maximum_bucket': bucket(upper, config),
        'review_required': bool(unknown), 'unknown_flags': unknown,
        'signals': signal_values, 'critical_overrides': critical, 'reasons': reasons,
    }


def score_contracts(records, config=None):
    if not records:
        raise ValueError('No risk records supplied.')
    if len({r['record_id'] for r in records}) != len(records):
        raise ValueError('Duplicate record IDs.')
    scored = [score_record(r, config) for r in records]
    groups = {}
    for row in scored:
        groups.setdefault(row['contract_id'], []).append(row)
    triage = []
    for contract, rows in sorted(groups.items()):
        if len({r['split'] for r in rows}) != 1:
            raise ValueError('A contract appears in multiple splits.')
        lower, upper = max(r['score_lower'] for r in rows), max(r['score_upper'] for r in rows)
        complete = not any(r['review_required'] for r in rows)
        triage.append({'contract_id': contract, 'score_lower': lower, 'score_upper': upper,
                      'score': lower if complete else None,
                      'bucket': bucket(lower, config or default_config()) if complete or any(r['critical_overrides'] for r in rows) else None,
                      'review_required': not complete, 'record_count': len(rows),
                      'coverage': 'supplied records only; no automatic full-contract completeness claim'})
    return scored, triage
