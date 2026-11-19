import unittest
from hybrid_integration.adapter import adapt

class AdapterTests(unittest.TestCase):
    def test_no_detections_never_implies_absence(self):
        p = {'smoke_test': False, 'contracts': [{'contract_id': 'x', 'split': 'test', 'detections': []}]}
        records, queue = adapt(p)
        self.assertTrue(all(v['status'] == 'unknown' for v in records[0]['assessments'].values()))
        self.assertTrue(queue[0]['review_required'])

    def test_smoke_test_rejected(self):
        with self.assertRaises(ValueError):
            adapt({'smoke_test': True, 'contracts': []})
