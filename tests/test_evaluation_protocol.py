import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import eval_protocol as ep


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixedModel:
    def __init__(self, scores, predictions=None):
        self.scores = scores
        self.predictions = predictions

    def score(self, users, candidates):
        return self.scores

    def predict(self, users, places):
        return self.predictions


def tiny_protocol():
    train = pd.DataFrame(
        [{"user_id": "u1", "place_id": "p1", "rating": 4.0}]
    )
    validation = pd.DataFrame(
        [{"user_id": "u1", "place_id": "p2", "rating": 5.0}]
    )
    return {
        "year": "2022",
        "train": train,
        "validation": validation,
        "test": validation.copy(),
        "cand": np.array(["p1", "p2", "p3"]),
        "global_mean": 4.0,
    }


class EvaluationProtocolTests(unittest.TestCase):
    def test_load_data_rejects_user_place_split_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "year": "2022",
                        "user_id": "u1",
                        "place_id": "p1",
                        "rating": 4,
                        "known_user_split": "train",
                    },
                    {
                        "year": "2022",
                        "user_id": "u1",
                        "place_id": "p1",
                        "rating": 5,
                        "known_user_split": "validation",
                    },
                ]
            ).to_csv(root / "interactions.csv", index=False)
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                ep.load_data("2022", root)

    def test_candidate_set_smaller_than_k_is_supported(self):
        metrics, predictions = ep.evaluate(
            tiny_protocol(), FixedModel([[0.2, 0.9, 0.1]]), "validation"
        )
        self.assertEqual(metrics["ranking_k_requested"], 10)
        self.assertEqual(metrics["ranking_k_effective"], 3)
        self.assertEqual(metrics["recall_at_10"], 1.0)
        self.assertEqual(metrics["ndcg_at_10"], 1.0)
        self.assertEqual(predictions.place_id.tolist(), ["p2", "p3"])

    def test_score_shape_and_values_are_checked(self):
        with self.assertRaisesRegex(ValueError, "score.*shape"):
            ep.evaluate(tiny_protocol(), FixedModel([[1.0, 2.0]]), "validation")
        with self.assertRaisesRegex(ValueError, "NaN"):
            ep.evaluate(
                tiny_protocol(), FixedModel([[0.0, np.nan, 1.0]]), "validation"
            )
        with self.assertRaisesRegex(ValueError, "\\+inf"):
            ep.evaluate(
                tiny_protocol(), FixedModel([[0.0, np.inf, 1.0]]), "validation"
            )

    def test_rating_prediction_shape_is_checked(self):
        with self.assertRaisesRegex(ValueError, "predict.*shape"):
            ep.evaluate(
                tiny_protocol(),
                FixedModel([[0.2, 0.9, 0.1]], predictions=np.array([])),
                "validation",
            )


class CheckDataTests(unittest.TestCase):
    def test_manifest_paths_and_utf8_names_are_portable(self):
        check_data = load_script("check_data")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            csv_path = data / "하위 폴더" / "자료.csv"
            csv_path.parent.mkdir(parents=True)
            csv_path.write_bytes("값\n1\n".encode("utf-8"))
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    [{"path": "하위 폴더/자료.csv", "bytes": csv_path.stat().st_size}],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self.assertEqual(check_data.check_manifest(data, manifest), [])


if __name__ == "__main__":
    unittest.main()
