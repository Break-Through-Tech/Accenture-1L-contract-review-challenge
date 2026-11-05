import unittest
import tempfile
from pathlib import Path
from copy import deepcopy
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.task4_analysis import analyze, validate_bundle, write_results


def bundle():
    return {"schema_version": 1, "synthetic": False, "run_description": "Unit-test fixtures, not CUAD results",
            "labels": ["A", "B"],
            "chunks": [
                {"contract_title": "C", "chunk_id": "1", "chunk_text": "Unless otherwise agreed, Section 2 controls."},
                {"contract_title": "C", "chunk_id": "2", "chunk_text": "=SUM(1,2)"},
                {"contract_title": "D", "chunk_id": "3", "chunk_text": "Ordinary language."}],
            "y_true": [[1,0], [0,1], [0,0]],
            "models": {"bert": {"predictions": [[0,1],[1,1],[0,0]],
                                "probabilities": [[.3,.7],[.8,.9],[.1,.2]]}}}


class TestTask4(unittest.TestCase):
    def test_exact_counts(self):
        out = analyze(bundle())
        a = out["per_category"].set_index("category")
        self.assertEqual((int(a.loc["A", "tp"]), int(a.loc["A", "fp"]), int(a.loc["A", "fn"])), (0,1,1))
        self.assertEqual((int(a.loc["B", "tp"]), int(a.loc["B", "fp"]), int(a.loc["B", "fn"])), (1,1,0))
        self.assertAlmostEqual(out["model_summary"].iloc[0]["micro_f1"], 0.4)
        self.assertEqual(len(out["review_samples"]), 3)

    def test_no_errors(self):
        b = bundle()
        b["models"]["bert"]["predictions"] = b["y_true"]
        self.assertEqual(len(analyze(b)["review_samples"]), 0)
        self.assertEqual(int(analyze(b)["per_category"]["fp"].sum()), 0)

    def test_shape_and_duplicate_rejection(self):
        b = bundle()
        b["models"]["bert"]["predictions"].pop()
        with self.assertRaises(ValueError):
            validate_bundle(b)
        b = bundle()
        b["chunks"][1]["chunk_id"] = "1"
        with self.assertRaises(ValueError):
            validate_bundle(b)

    def test_synthetic_guard(self):
        b = bundle()
        b["synthetic"] = True
        with self.assertRaises(ValueError):
            analyze(b)
        self.assertEqual(len(analyze(b, allow_synthetic=True)["per_category"]), 2)

    def test_probability_guard(self):
        b = bundle()
        b["models"]["bert"]["probabilities"][0][0] = 1.2
        with self.assertRaises(ValueError):
            validate_bundle(b)

    def test_missing_probability_and_empty_support(self):
        b = bundle()
        del b["models"]["bert"]["probabilities"]
        b["y_true"] = [[1,0], [0,0], [0,0]]
        df = analyze(b)["per_category"].set_index("category")
        self.assertEqual(int(df.loc["B", "support"]), 0)
        self.assertTrue(df.loc["B", "recall"] != df.loc["B", "recall"] or df.loc["B", "recall"] is None)

    def test_exports_and_csv_injection(self):
        with tempfile.TemporaryDirectory() as p:
            out = write_results(bundle(), p)
            self.assertTrue((Path(p)/"run_metadata.json").exists())
            self.assertTrue((Path(p)/"error_patterns.csv").exists())
            self.assertTrue(any(str(x).startswith("'=") for x in out["review_samples"]["chunk_text"]))


if __name__ == "__main__":
    unittest.main()
