"""Read-only diagnosis of Part A detections vs Phase 2 assessor rules."""
import argparse, json, re
from collections import Counter
from pathlib import Path
from hybrid_integration.assessor import RULES, CATEGORIES


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--detections',default='outputs/pipeline/detections_test.json')
    p.add_argument('--samples',type=int,default=3)
    args=p.parse_args()
    data=json.loads(Path(args.detections).read_text())
    contracts=data['contracts']
    detections=[(c['contract_id'],d) for c in contracts for d in c.get('detections',[])]
    print('Contracts:',len(contracts),'Detections:',len(detections),'Smoke test:',data.get('smoke_test'))
    print('Top categories:',Counter(d.get('category') for _,d in detections).most_common(12))
    for flag,patterns in RULES.items():
        eligible=[(cid,d) for cid,d in detections if not CATEGORIES[flag] or d.get('category') in CATEGORIES[flag]]
        matched=[(cid,d) for cid,d in eligible if any(re.search(p,d.get('text',''),re.I|re.S) for p in patterns)]
        print(f'\n{flag}: eligible detections={len(eligible)}, regex matches={len(matched)}')
        for cid,d in eligible[:args.samples]:
            print('  category:',d.get('category'),'contract:',cid[:65])
            print('  excerpt:',re.sub(r'\s+',' ',d.get('text',''))[:320])
    print('\nDiagnostic only: no evidence of a risk is not proof of safety or absence.')

if __name__=='__main__': main()
