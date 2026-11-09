import argparse
import csv
import hashlib
import json
from pathlib import Path
from .engine import FLAGS, default_config, score_contracts
from .calibration import calibrate, evaluate_holdout


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Explainable four-signal risk triage and advisor calibration')
    parser.add_argument('command', choices=['score', 'calibrate', 'evaluate', 'prepare-advisor'])
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    parser.add_argument('--limit', type=int, default=25)
    args = parser.parse_args()
    if args.command == 'prepare-advisor':
        if args.limit < 1:
            parser.error('--limit must be positive')
        categories = {'Cap On Liability','Uncapped Liability','Termination For Convenience',
                      'Renewal Term','Notice Period To Terminate Renewal','Ip Ownership Assignment'}
        grouped = {}
        csv.field_size_limit(10_000_000)
        with open(args.input) as f:
            for row in csv.DictReader(f):
                category = row['category']
                if category in categories and row['clause_text'].strip() and row['is_impossible'].lower() == 'false':
                    grouped.setdefault(category, []).append(row)
        selected, seen = [], set()
        for i in range(max((len(g) for g in grouped.values()),default=0)):
            for category, group in sorted(grouped.items()):
                if i >= len(group):
                    continue
                row = group[i]
                key = (row['contract_title'],row['clause_text'])
                if key in seen:
                    continue
                seen.add(key)
                token = hashlib.sha256((row['contract_title']+'|'+row['answer_start']+'|'+category).encode()).hexdigest()[:16]
                selected.append({'record_id':token, 'contract_id':row['contract_title'],
                    'category':category, 'text':row['clause_text'], 'split':'validation',
                    'source':f"{args.input}:answer_start={row['answer_start']}",
                    'advisor_rating':None,
                    'assessments':{flag:{'status':'unknown','rationale':'','reviewer':''} for flag in FLAGS}})
                if len(selected) == args.limit:
                    break
            if len(selected) == args.limit:
                break
        if not selected:
            raise ValueError('No relevant evidence found.')
        save(args.output, selected)
        print(f'Saved {len(selected)} real excerpts for review. No advisor ratings or risk assessments inferred.')
        return
    records = load(args.input)
    config = load(args.config) if args.config else default_config()
    config = config.get('config', config)
    if args.command == 'score':
        clauses, contracts = score_contracts(records,config)
        result = {'schema_version':1,'config':config,'clauses':clauses,'contracts':contracts}
    elif args.command == 'calibrate':
        fitted, report = calibrate(records,config)
        result = {'config':fitted,'report':report}
    else:
        if not args.config:
            parser.error('evaluate requires --config pointing to the frozen fitted config JSON')
        result = evaluate_holdout(records,config)
    save(args.output,result)
    print('Saved',args.output)


if __name__ == '__main__':
    main()
