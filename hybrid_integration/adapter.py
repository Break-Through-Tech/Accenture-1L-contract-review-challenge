"""Conservative Part A -> Part B adapter. No legal risk flag is inferred from clause presence alone."""
import json
from pathlib import Path
from collections import defaultdict

CATEGORY_HINTS = {
    'missing_liability': ('Cap On Liability', 'Uncapped Liability'),
    'missing_indemnification': (),
    'missing_ip_ownership': ('Ip Ownership Assignment',),
    'uncapped_liability': ('Uncapped Liability', 'Cap On Liability'),
    'open_ended_indemnification': (),
    'one_sided_termination': ('Termination For Convenience',),
    'renewal_without_notice': ('Renewal Term', 'Notice Period To Terminate Renewal'),
    'ambiguous_ip_ownership': ('Ip Ownership Assignment',),
}
PRIORITY_FLAGS = {'uncapped_liability', 'missing_indemnification'}


def adapt(payload):
    from risk_scoring.engine import FLAGS
    if payload.get('smoke_test') is not False:
        raise ValueError('Refusing smoke-test or unverified input')
    if not isinstance(payload.get('contracts'), list):
        raise ValueError('Expected contracts list')
    records, queue = [], []
    for contract in payload['contracts']:
        cid = contract['contract_id']
        split = contract['split']
        if split not in ('train', 'validation', 'test', 'synthetic'):
            raise ValueError(f'Unsupported split: {split}')
        detections = contract['detections']
        by_category = defaultdict(list)
        for d in detections:
            by_category[d['category']].append(d)
        # One full-contract assessment record; detections are *hints*, not proof of absence.
        # The engine requires nonempty text. A contextual evidence digest is not the full contract.
        snippets = []
        for d in detections:
            excerpt = d.get('text', '').strip()
            if excerpt:
                snippets.append(f"[{d['category']} | chunk {d.get('chunk_id', '?')}] {excerpt[:600]}")
        context = '\n\n'.join(snippets[:30]) or '[No clause detections; full contract must be inspected]'
        assessments = {f: {'status': 'unknown'} for f in FLAGS}
        record = {'record_id': f'{split}:{cid}', 'contract_id': cid,
                  'category': 'Full-contract risk triage', 'text': context,
                  'source': 'Part A detection excerpts; NOT full-contract text',
                  'split': split, 'assessments': assessments}
        records.append(record)
        hints = {}
        for flag, categories in CATEGORY_HINTS.items():
            matching = [d for cat in categories for d in by_category.get(cat, [])]
            hints[flag] = {'status': 'needs_human_assessment',
                           'matching_detection_count': len(matching),
                           'evidence_candidates': [{'category': d['category'], 'chunk_id': d.get('chunk_id'),
                                                    'probability': d.get('probability'),
                                                    'excerpt': d.get('text', '')[:350]}
                                                   for d in matching[:3]]}
        queue.append({'record_id': record['record_id'], 'contract_id': cid,
                      'priority': 'high' if any(hints[f]['matching_detection_count'] for f in PRIORITY_FLAGS) else 'standard',
                      'review_required': True, 'full_contract_available': False,
                      'warning': 'Detection excerpts cannot establish clause absence or full-contract coverage.',
                      'flags': hints})
    return records, queue


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--records', required=True)
    parser.add_argument('--queue', required=True)
    args = parser.parse_args()
    records, queue = adapt(json.loads(Path(args.input).read_text()))
    for path, value in ((args.records, records), (args.queue, queue)):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    print(f'Prepared {len(records)} records and {len(queue)} review items (all provisional).')


if __name__ == '__main__':
    main()
