"""Generate reviewer-ready CSVs and handoff notes from pipeline JSON outputs."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_json(path):
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def write_csv(path, columns, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def joined(value):
    if not value:
        return ''
    if isinstance(value, list):
        return '; '.join(str(v) for v in value)
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--triage', type=Path, default=Path('outputs/teammate_evaluation/auto_triage_test.json'))
    parser.add_argument('--validation', type=Path, default=Path('outputs/teammate_evaluation/validation_summary.json'))
    parser.add_argument('--out-dir', type=Path, default=Path("outputs/handoff_milestone3_task2"))
    args = parser.parse_args()

    triage = load_json(args.triage)
    validation = load_json(args.validation)
    contracts = triage.get('contracts')
    if not isinstance(contracts, list):
        raise ValueError('Triage JSON must contain a contracts list')
    if not validation.get('structural_validation_passed'):
        raise ValueError('Structural validation has not passed; do not generate handoff')
    if validation.get('total_contracts') != len(contracts):
        raise ValueError('Validation and triage contract counts differ')
    if len({c.get('contract_id') for c in contracts}) != len(contracts):
        raise ValueError('Duplicate contract_id found in triage data')
    actual_levels = Counter(
        'None' if c.get('risk_level') is None else str(c.get('risk_level'))
        for c in contracts
    )
    expected_levels = Counter(validation.get('risk_levels', {}))
    if actual_levels != expected_levels:
        raise ValueError(
            f'Risk level counts do not match validation summary: '
            f'actual={dict(actual_levels)}, expected={dict(expected_levels)}'
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for contract in contracts:
        findings = contract.get('findings') or []
        if not isinstance(findings, list):
            raise ValueError('Expected findings list')
        finding_flags = [f.get('flag', '') for f in findings if isinstance(f, dict)]
        # A label can be displayed for triage, but is not advisor-validated.
        level = contract.get('risk_level')
        status = contract.get('status', '')
        report.append({
            'contract_id': contract.get('contract_id', ''),
            'split': contract.get('split', ''),
            'risk_level': level if level is not None else 'Needs review',
            'score': contract.get('score') if contract.get('score') is not None else '',
            'score_lower': contract.get('score_lower', ''),
            'score_upper': contract.get('score_upper', ''),
            'assessment_status': status,
            'review_required': status == 'human_review' or level is None,
            'critical_overrides': joined(contract.get('critical_overrides')),
            'unknown_flags': joined(contract.get('unknown_flags')),
            'review_reason': contract.get('review_reason') or '',
            'finding_flags': joined(finding_flags),
            'finding_explanations': ' | '.join(
                f"{f.get('flag', '')}: {f.get('why', '')}" for f in findings if isinstance(f, dict)
            ),
            'classification_source': 'automated rules + model detections; not advisor-validated',
        })

    audit = validation.get('critical_override_audit', [])
    if not isinstance(audit, list) or len(audit) != validation.get('critical_override_contracts'):
        raise ValueError('Critical override audit count mismatch')
    contract_ids = {c['contract_id'] for c in contracts}
    if any(a.get('contract_id') not in contract_ids for a in audit):
        raise ValueError('Audit contains contract IDs missing from triage')
    audit_rows = [{
        'contract_id': a.get('contract_id', ''),
        'risk_level': a.get('risk_level', ''),
        'critical_overrides': joined(a.get('critical_overrides')),
        'unknown_flags': joined(a.get('unknown_flags')),
        'score_lower': a.get('score_lower', ''),
        'score_upper': a.get('score_upper', ''),
        'requires_evidence_audit': a.get('requires_evidence_audit', True),
        'reviewer_decision': '',
        'reviewer_notes': '',
    } for a in audit]

    write_csv(args.out_dir / 'contract_risk_report.csv', list(report[0]) if report else [], report)
    write_csv(args.out_dir / 'critical_override_audit.csv', list(audit_rows[0]) if audit_rows else [], audit_rows)

    counts = Counter(c.get('risk_level') for c in contracts)
    unresolved = counts.get(None, 0)
    notes = f'''# Accenture contract risk pipeline — Task #2 handoff

## What was delivered

The integrated pipeline uses Part A contract clause detections and full-contract text to derive eight rule-based risk flags, then calls the existing `risk_scoring` engine. Its output includes contract-level risk triage, score ranges, explanations, and human-review routing.

## Input and output

- Input: `{args.triage.as_posix()}`
- Structural validation: `{args.validation.as_posix()}`
- Contract report: `contract_risk_report.csv`
- Critical-override review: `critical_override_audit.csv`

## Current test-split results

- Contracts: {len(contracts)}
- High: {counts.get('High', 0)}
- Medium: {counts.get('Medium', 0)}
- Low: {counts.get('Low', 0)}
- Unresolved / needs review: {unresolved}
- Critical-override cases requiring evidence audit: {len(audit_rows)}
- Structural validation: passed
- Accuracy against advisor labels: NOT VALIDATED

## Instructions for Task #3

1. Use advisor-labeled contracts to independently assess whether each automated risk flag and contract-level risk label is correct.
2. Prioritize the critical-override audit, especially `missing_indemnification` and `uncapped_liability` findings. Review full contract context, exceptions, and whether the protection is appropriate to the contract type.
3. Evaluate false positives and false negatives, including samples from the Low, Medium, High, and unresolved groups. Do not assume that a missing detection proves that a clause is absent.
4. Record decisions and supporting excerpts in the audit CSV; do not treat `reviewer=auto:rules-v1` as a human approval.
5. Calculate classification accuracy and per-class precision/recall/F1 where reference labels permit. Document ambiguous labels and any disagreements.
6. Re-run the pipeline and structural validation after any rule or threshold changes. Avoid using the held-out test split for tuning.

## Important limitations

The risk categories are automated screening outputs, not legal conclusions. A `High` label is not necessarily a verified High risk. Scores may be missing where flags remain unknown; use score bounds rather than inventing an exact score. Structural validation confirms data consistency only, not model accuracy or evidence correctness.
'''
    (args.out_dir / 'HANDOFF.md').write_text(notes, encoding='utf-8')
    print(f'Generated 3 files in {args.out_dir} for {len(contracts)} contracts; {len(audit_rows)} critical audits.')


if __name__ == '__main__':
    main()
