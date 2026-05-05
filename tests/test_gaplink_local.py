import unittest
from pathlib import Path

from gaplink_local import evaluate, generate_candidates, load_restaurants, run
from llm_matcher import parse_decision


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "relational-dataset" / "fodors-zagats"


class GaplinkLocalTests(unittest.TestCase):
    def test_generates_candidates_from_bundled_data(self):
        left = load_restaurants(DATASET / "tableA.csv", "A")
        right = load_restaurants(DATASET / "tableB.csv", "B", id_offset=len(left))

        candidates = generate_candidates(left, right)

        self.assertTrue(candidates)
        self.assertGreaterEqual(candidates[0].score, candidates[-1].score)
        self.assertTrue(all(candidate.left.source == "A" for candidate in candidates))
        self.assertTrue(all(candidate.right.source == "B" for candidate in candidates))

    def test_evaluate_counts_precision_recall_f1(self):
        metrics = evaluate({(1, 2), (3, 4)}, {(1, 2), (5, 6)})

        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)

    def test_local_run_writes_outputs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metrics = run(dataset_dir=DATASET, output_dir=tmp_path, threshold=0.48)

            self.assertGreater(metrics["candidate_pairs"], 0)
            self.assertGreater(metrics["predicted_matches"], 0)
            self.assertGreater(metrics["f1"], 0)
            self.assertTrue((tmp_path / "query_results.csv").exists())
            self.assertTrue((tmp_path / "filtered_pattern.txt").exists())
            self.assertTrue((tmp_path / "blocks.txt").exists())
            self.assertTrue((tmp_path / "results.txt").exists())
            self.assertTrue((tmp_path / "metrics.txt").exists())

    def test_llm_decision_parser_accepts_json(self):
        decision = parse_decision('{"match": true, "confidence": 0.87, "reason": "same phone"}')

        self.assertTrue(decision.match)
        self.assertEqual(decision.confidence, 0.87)
        self.assertEqual(decision.reason, "same phone")


if __name__ == "__main__":
    unittest.main()
