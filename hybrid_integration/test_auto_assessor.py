import unittest

from hybrid_integration.auto_assessor import assess_contract, build_records, triage
from risk_scoring.engine import FLAGS, score_contracts

CAPPED = ("Each party shall indemnify and hold harmless the other party from third-party claims. "
          "In no event shall either party's aggregate liability exceed the fees paid in the prior twelve months. "
          "Either party may terminate this Agreement for convenience upon thirty (30) days' written notice.")
CARVED = ("Each party shall indemnify the other party against third-party claims. "
          "EXCEPT FOR A PARTY'S INDEMNIFICATION OBLIGATIONS UNDER SECTION 9, IN NO EVENT SHALL EITHER PARTY'S LIABILITY EXCEED $100,000.")
BARE = "The Supplier shall deliver the goods. The Customer shall pay the invoice within thirty days."


def contract(cid, dets=()):
    return {'contract_id': cid, 'split': 'test', 'chunks_scanned': 3, 'detections': list(dets)}


def det(category, text, p=0.95, thr=0.5):
    return {'category': category, 'chunk_id': 0, 'text': text, 'probability': p, 'threshold': thr}


class RuleTests(unittest.TestCase):
    def test_capped_mutual_contract_is_resolved_low(self):
        a = assess_contract(contract('a'), CAPPED)
        self.assertEqual(a['missing_liability']['status'], 'absent')
        self.assertEqual(a['uncapped_liability']['status'], 'absent')
        self.assertEqual(a['missing_indemnification']['status'], 'absent')
        self.assertEqual(a['one_sided_termination']['status'], 'absent')
        self.assertEqual(a['renewal_without_notice']['status'], 'not_applicable')

    def test_carve_out_for_indemnity_is_uncapped_and_open_ended(self):
        a = assess_contract(contract('b'), CARVED)
        self.assertEqual(a['uncapped_liability']['status'], 'present')
        self.assertEqual(a['open_ended_indemnification']['status'], 'present')

    def test_missing_protections_require_full_contract_scope(self):
        a = assess_contract(contract('c'), BARE)
        self.assertEqual(a['missing_liability']['status'], 'present')
        self.assertEqual(a['missing_liability']['scope'], 'full_contract')
        self.assertEqual(a['missing_indemnification']['status'], 'present')
        # engine forbids missing_X and the X-characteristic both present
        self.assertEqual(a['uncapped_liability']['status'], 'not_applicable')
        self.assertEqual(a['open_ended_indemnification']['status'], 'not_applicable')

    def test_model_label_conflicting_with_text_goes_to_review(self):
        text = "Vendor's total liability under this Agreement shall not exceed the fees paid."
        a = assess_contract(contract('d', [det('Uncapped Liability', text, p=0.97, thr=0.9)]), text)
        self.assertEqual(a['uncapped_liability']['status'], 'unknown')

    def test_one_way_termination_and_survival_clause(self):
        a = assess_contract(contract('e'), 'Acme may terminate this Agreement at any time for convenience.')
        self.assertEqual(a['one_sided_termination']['status'], 'present')
        a = assess_contract(contract('f'), 'Sections 4 and 5 shall survive termination of this Agreement for any reason.')
        self.assertEqual(a['one_sided_termination']['status'], 'absent')

    def test_auto_renewal_with_and_without_notice(self):
        a = assess_contract(contract('g'), 'This Agreement shall automatically renew for successive one-year terms.')
        self.assertEqual(a['renewal_without_notice']['status'], 'present')
        a = assess_contract(contract('h'), 'This Agreement shall automatically renew for successive one-year terms unless either party gives sixty (60) days prior written notice of non-renewal.')
        self.assertEqual(a['renewal_without_notice']['status'], 'absent')

    def test_joint_venture_disclaimer_is_not_joint_ip(self):
        text = ('All intellectual property developed hereunder shall be the sole property of Acme. '
                'Nothing in this Agreement shall create a partnership, joint venture or co-ownership. '
                'Acme owns the intellectual property and patents.')
        a = assess_contract(contract('i'), text)
        self.assertNotEqual(a['ambiguous_ip_ownership']['status'], 'present')

    def test_every_flag_has_provenance(self):
        for text in (CAPPED, CARVED, BARE):
            a = assess_contract(contract('j'), text)
            self.assertEqual(set(a), set(FLAGS))
            for flag, v in a.items():
                self.assertTrue(v['reviewer'].startswith('auto:'), flag)
                self.assertTrue(v['rationale'], flag)


class TriageTests(unittest.TestCase):
    def run_pipeline(self, texts):
        payload = {'smoke_test': False, 'contracts': [contract(cid) for cid in texts]}
        records = build_records(payload, texts)
        scored, _ = score_contracts(records)
        return {r['contract_id']: r for r in triage(scored, records)}

    def test_records_pass_engine_validation_and_get_buckets(self):
        rows = self.run_pipeline({'capped': CAPPED, 'carved': CARVED, 'bare': BARE})
        self.assertEqual(rows['capped']['risk_level'], 'Low')
        self.assertEqual(rows['carved']['risk_level'], 'High')
        self.assertEqual(rows['bare']['risk_level'], 'High')  # missing indemnification is a critical override
        self.assertTrue(all(r['status'].startswith('auto_classified') for r in rows.values()))

    def test_smoke_test_rejected(self):
        with self.assertRaises(ValueError):
            build_records({'smoke_test': True, 'contracts': []}, {})


if __name__ == '__main__':
    unittest.main()
