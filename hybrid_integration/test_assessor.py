import unittest
from hybrid_integration.assessor import assess

FLAGS = ('missing_liability', 'missing_indemnification', 'missing_ip_ownership',
         'uncapped_liability', 'open_ended_indemnification', 'one_sided_termination',
         'renewal_without_notice', 'ambiguous_ip_ownership')


def fixture(text, category='Uncapped Liability'):
    payload = {'smoke_test': False, 'contracts': [{'contract_id': 'a', 'split': 'test',
        'detections': [{'category': category, 'text': text, 'chunk_id': 0, 'probability': .92}]}]}
    records = [{'record_id': 'test:a', 'contract_id': 'a', 'split': 'test',
                'assessments': {f: {'status': 'unknown'} for f in FLAGS}}]
    queue = [{'record_id': 'test:a', 'priority': 'standard'}]
    return payload, records, queue


class AssessorTests(unittest.TestCase):
    def test_explicit_candidate_remains_unconfirmed(self):
        records, queue = assess(*fixture('The parties agree to unlimited liability for damages.'))
        self.assertEqual(queue[0]['automated_candidates'][0]['flag'], 'uncapped_liability')
        self.assertEqual(records[0]['assessments']['uncapped_liability']['status'], 'unknown')

    def test_no_match_does_not_imply_absence(self):
        records, queue = assess(*fixture('Liability shall not exceed $100,000.'))
        self.assertEqual(queue[0]['candidate_count'], 0)
        self.assertTrue(all(a['status'] == 'unknown' for a in records[0]['assessments'].values()))

    def test_smoke_test_rejected(self):
        payload, records, queue = fixture('Unlimited liability')
        payload['smoke_test'] = True
        with self.assertRaises(ValueError):
            assess(payload, records, queue)

    def test_split_mismatch_rejected(self):
        payload, records, queue = fixture('Unlimited liability')
        records[0]['split'] = 'validation'
        with self.assertRaises(ValueError):
            assess(payload, records, queue)


if __name__ == '__main__':
    unittest.main()

def test_conflicting_indemnification_evidence():
    from hybrid_integration.assessor import classify_candidate

    evidence = (
        "The parties shall indemnify against any and all losses. "
        "Each party's liability for damages hereunder "
        "shall not exceed $1,000,000."
    )

    result = classify_candidate(
        "open_ended_indemnification",
        "indemnify against any and all losses",
        evidence
    )

    assert result == "conflicting_evidence"