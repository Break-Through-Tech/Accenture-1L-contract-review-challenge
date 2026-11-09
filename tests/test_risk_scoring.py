import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from risk_scoring.engine import FLAGS, SIGNALS, default_config, score_record, score_contracts
from risk_scoring.calibration import calibrate, evaluate_holdout, metrics


def record(i=0):
    return {'record_id':str(i), 'contract_id':f'contract-{i}', 'category':'Synthetic example',
            'text':'Liability is unlimited. Termination is unilateral.', 'source':'synthetic unit test',
            'split':'validation', 'advisor_rating':1,
            'assessments':{f:{'status':'absent','rationale':'Synthetic test assessment only','reviewer':'test'} for f in FLAGS}}


def flag(r, name):
    r['assessments'][name].update(status='present', evidence=r['text'], scope='full_contract')
    return r


class RiskTests(unittest.TestCase):
    def test_clean(self):
        result = score_record(record())
        self.assertEqual(result['score'],0)
        self.assertEqual(result['bucket'],'Low')
    def test_critical_overrides(self):
        for f in ['uncapped_liability','missing_indemnification']:
            result=score_record(flag(record(),f))
            self.assertEqual(result['bucket'],'High')
            self.assertEqual(result['critical_overrides'],[f])
    def test_missing_floor(self):
        config=default_config();config['weights']['missing_protections']=0
        self.assertEqual(score_record(flag(record(),'missing_ip_ownership'),config)['bucket'],'Medium')
    def test_unknown_not_low(self):
        r=record()
        for a in r['assessments'].values():a['status']='unknown'
        result=score_record(r)
        self.assertIsNone(result['score']);self.assertIsNone(result['bucket'])
        self.assertEqual((result['score_lower'],result['score_upper']),(0,100))
        flag(r,'uncapped_liability')
        self.assertEqual(score_record(r)['bucket'],'High')
        self.assertTrue(score_record(r)['review_required'])
    def test_evidence_and_missing_scope(self):
        r=flag(record(),'uncapped_liability');r['assessments']['uncapped_liability']['evidence']='not in excerpt'
        with self.assertRaises(ValueError):score_record(r)
        r=flag(record(),'missing_liability');r['assessments']['missing_liability']['scope']='chunk'
        with self.assertRaises(ValueError):score_record(r)
    def test_conflicts(self):
        with self.assertRaises(ValueError):score_record(flag(flag(record(),'missing_liability'),'uncapped_liability'))
    def test_invalid_config(self):
        for weights in [{s:0 for s in SIGNALS},{s:float('nan') for s in SIGNALS}]:
            c=default_config();c['weights']=weights
            with self.assertRaises(ValueError):score_record(record(),c)
    def test_contract_max_and_duplicates(self):
        a,b=record(),flag(record(1),'uncapped_liability');b['contract_id']=a['contract_id']
        _,triage=score_contracts([a,b])
        self.assertEqual(triage[0]['bucket'],'High')
        with self.assertRaises(ValueError):score_contracts([a,a])
    def test_calibration_and_holdout(self):
        records=[record(i) for i in range(3)]
        flag(records[1],'one_sided_termination');records[1]['advisor_rating']=3
        flag(records[2],'uncapped_liability');records[2]['advisor_rating']=5
        config,report=calibrate(records)
        self.assertEqual(report['candidates_evaluated'],144)
        self.assertEqual(report['after_in_sample']['bucket_agreement'],1)
        test=record(99);test['split']='test'
        self.assertEqual(evaluate_holdout([test],config)['held_out_metrics']['bucket_agreement'],1)
        test['contract_id']=records[0]['contract_id']
        with self.assertRaises(ValueError):evaluate_holdout([test],config)
        records[0]['split']='test'
        with self.assertRaises(ValueError):calibrate(records)
    def test_unknown_and_invalid_ratings_cannot_calibrate(self):
        records=[record(i) for i in range(3)]
        records[2]['advisor_rating']=5;records[1]['advisor_rating']=3
        records[0]['assessments'][FLAGS[0]]['status']='unknown'
        with self.assertRaises(ValueError):calibrate(records)
        records[0]=record();records[0]['advisor_rating']=6
        with self.assertRaises(ValueError):calibrate(records)
    def test_cli_fitted_artifact_roundtrip(self):
        rows=[record(i) for i in range(3)]
        flag(rows[1],'one_sided_termination');rows[1]['advisor_rating']=3
        flag(rows[2],'uncapped_liability');rows[2]['advisor_rating']=5
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            (d/'development.json').write_text(json.dumps(rows))
            subprocess.run([sys.executable,'-m','risk_scoring','calibrate','--input',str(d/'development.json'),
                            '--output',str(d/'fitted.json')],check=True,capture_output=True)
            heldout=record(99);heldout['split']='test'
            (d/'test.json').write_text(json.dumps([heldout]))
            subprocess.run([sys.executable,'-m','risk_scoring','evaluate','--input',str(d/'test.json'),
                            '--config',str(d/'fitted.json'),'--output',str(d/'report.json')],check=True,capture_output=True)
            self.assertEqual(json.loads((d/'report.json').read_text())['held_out_metrics']['bucket_agreement'],1)
    def test_pending_packet_remains_unscored(self):
        rows=json.loads(Path('examples/advisor_review_pending.json').read_text())
        result,_=score_contracts(rows)
        self.assertEqual(len(result),25)
        self.assertTrue(all(r['score'] is None and r['bucket'] is None for r in result))
        self.assertTrue(all(r['advisor_rating'] is None for r in rows))
    def test_undefined_correlation(self):
        result=metrics([record(),record(1)],default_config())
        self.assertIsNone(result['spearman']);self.assertIsNone(result['high_risk_recall'])


if __name__ == '__main__':
    unittest.main()
