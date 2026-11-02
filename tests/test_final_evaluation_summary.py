import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_final_evaluation_summary import verify_common_context, verify_hybrid_endpoint


def bundle(model="cbf_region", mode="warm", dirty=False, ndcg=0.1):
    return {
        "meta": {
            "model": model,
            "year": "2022",
            "split": "test",
            "git": {"commit": "abc", "dirty": dirty},
            "protocol": f"v1_1/known_user_2022/{mode}",
            "candidate_mode": mode,
            "candidate_sha256": "candidate",
            "evaluation_pairs_sha256": "pairs",
            "positive_users_sha256": "positive",
            "warm_positive_users_sha256": "warm-positive",
            "data_sha256": {"interactions.csv": "data"},
        },
        "metrics": {
            "eval_users": 10,
            "ranked_users": 8,
            "relevant_total": 8,
            "relevant_warm": 5,
            "relevant_cold": 3,
            "candidates": 20,
            "train_pairs": 30,
            "eval_pairs": 10,
            "protocol_version": "1.1",
            "recall_at_10": 0.2,
            "ndcg_at_10": ndcg,
            "recall_at_10_warm": 0.3,
            "ndcg_at_10_warm": 0.15,
        },
    }


class FinalEvaluationSummaryTests(unittest.TestCase):
    def test_accepts_same_clean_context(self):
        first = bundle()
        second = bundle(model="hybrid")
        signature = verify_common_context([first, second], "warm")
        self.assertEqual(signature["git_commit"], "abc")

    def test_rejects_different_candidate_hash(self):
        first = bundle()
        second = bundle(model="hybrid")
        second["meta"]["candidate_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "candidate_sha256"):
            verify_common_context([first, second], "warm")

    def test_rejects_dirty_context(self):
        with self.assertRaisesRegex(ValueError, "clean Git checkout"):
            verify_common_context([bundle(dirty=True)], "warm")

    def test_alpha_zero_requires_exact_ranking_endpoint(self):
        cbf = bundle()
        hybrid = bundle(model="hybrid")
        verify_hybrid_endpoint(cbf, hybrid, 0)
        hybrid["metrics"]["ndcg_at_10"] = 0.11
        with self.assertRaisesRegex(ValueError, "differs from CBF"):
            verify_hybrid_endpoint(cbf, hybrid, 0)


if __name__ == "__main__":
    unittest.main()
