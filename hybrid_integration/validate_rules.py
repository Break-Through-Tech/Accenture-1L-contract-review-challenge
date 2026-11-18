"""Check auto_assessor rules against CUAD expert annotations (train/validation only).

Builds "oracle" detections from the gold clause spans (probability 1.0), runs the same rules,
and compares each flag decision with a gold proxy derived from which CUAD categories
the experts highlighted. Use this to tune rules; never tune on test.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from hybrid_integration.auto_assessor import SPLIT_FILES, assess_contract, clean_contract_text, load_contract_texts

# flag -> (gold categories, meaning of "category present" for the flag)
PROXIES = {
    'missing_liability': (('Cap On Liability', 'Uncapped Liability'), 'absent'),
    'uncapped_liability': (('Uncapped Liability',), 'present'),
    'one_sided_termination': (('Termination For Convenience',), 'has_clause'),
    'renewal_without_notice': (('Renewal Term', 'Notice Period To Terminate Renewal'), 'has_clause'),
    'missing_ip_ownership': (('Ip Ownership Assignment', 'Joint Ip Ownership'), 'absent'),
    'ambiguous_ip_ownership': (('Joint Ip Ownership',), 'present'),
}


def oracle_payload(split, texts, window=700, repo=Path('.')):
    ann = pd.read_csv(repo / 'notebooks' / f'{SPLIT_FILES[split]}_annotations.csv')
    ann = ann[ann['is_impossible'].astype(str).str.lower() == 'false']
    gold = defaultdict(set)
    dets = defaultdict(list)
    for row in ann.itertuples():
        text = texts.get(row.contract_title)
        if text is None or not isinstance(row.clause_text, str):
            continue
        gold[row.contract_title].add(row.category)
        clause = clean_contract_text(row.clause_text)
        i = text.find(clause[:200])
        if i < 0:
            continue
        s, e = max(0, i - window), min(len(text), i + len(clause) + window)
        dets[row.contract_title].append({'category': row.category, 'chunk_id': i, 'char_start': s, 'char_end': e,
                                         'text': text[s:e], 'probability': 1.0, 'threshold': 0.5})
    contracts = [{'contract_id': cid, 'split': split, 'chunks_scanned': max(1, len(texts[cid]) // 2000),
                  'detections': dets.get(cid, [])} for cid in texts]
    return contracts, gold


def evaluate(split='validation', repo=Path('.'), show=3):
    texts = load_contract_texts(split, repo)
    contracts, gold = oracle_payload(split, texts, repo=repo)
    tables = {f: Counter() for f in PROXIES}
    misses = defaultdict(list)
    for c in contracts:
        a = assess_contract(c, texts[c['contract_id']])
        g = gold.get(c['contract_id'], set())
        for flag, (cats, meaning) in PROXIES.items():
            has = bool(g & set(cats))
            status = a[flag]['status']
            tables[flag][(has, status)] += 1
            wrong = (meaning == 'absent' and ((has and status == 'present') or (not has and status == 'absent'))) or \
                    (meaning == 'present' and ((has and status == 'absent') or (not has and status == 'present'))) or \
                    (meaning == 'has_clause' and not has and status == 'present')
            if wrong:
                misses[flag].append((c['contract_id'][:70], status, a[flag]['rationale'][:160], (a[flag].get('evidence') or '')[:200]))
    report = {}
    for flag, t in tables.items():
        cats, meaning = PROXIES[flag]
        report[flag] = {'gold_proxy': f"{'/'.join(cats)} highlighted => {meaning}",
                        'with_gold_category': {k[1]: v for k, v in t.items() if k[0]},
                        'without_gold_category': {k[1]: v for k, v in t.items() if not k[0]},
                        'contradictions': len(misses[flag])}
    return report, misses


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='validation', choices=['train', 'validation'])
    p.add_argument('--show', type=int, default=3)
    args = p.parse_args()
    report, misses = evaluate(args.split)
    print(json.dumps(report, indent=2))
    for flag, rows in misses.items():
        print(f'\n## {flag}: {len(rows)} contradictions')
        for r in rows[:args.show]:
            print('  ', r)


if __name__ == '__main__':
    main()
