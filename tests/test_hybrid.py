import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from datemate.hybrid import combine_scores, couple_recommendations
from run_hybrid_real import Hybrid


def row(user, place, score):
    return dict(user_id=user, place_id=place, score=score,
                year='2022', split='validation')


class StaticModel:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def score(self, users, cand):
        return self.scores

    def predict(self, users, places):
        return None


class HybridTests(unittest.TestCase):
    def test_fallback_and_endpoints(self):
        cbf = [row('a', 'p', 0), row('a', 'q', 1)]
        cf = [row('a', 'p', 5)]
        scores = combine_scores(cbf, cf, 1)
        self.assertEqual([r['score'] for r in scores], [1, 1])
        self.assertEqual(scores[1]['score_source'], 'cbf_fallback')
        self.assertEqual(combine_scores(cbf, cf, 0)[0]['score'], .5)

    def test_real_hybrid_alpha_zero_is_exact_cbf_endpoint(self):
        cbf_scores = np.array([[0.25, np.nextafter(0.25, 1.0), 0.75]])
        cf_scores = np.array([[5.0, 1.0, -np.inf]])
        model = Hybrid(StaticModel(cbf_scores), StaticModel(cf_scores), 0)
        got = model.score(np.array(['u']), np.array(['a', 'b', 'c']))
        np.testing.assert_array_equal(got, cbf_scores)

    def test_real_hybrid_uses_cbf_for_unsupported_cf(self):
        cbf_scores = np.array([[0.2, 0.8]])
        cf_scores = np.array([[5.0, -np.inf]])
        got = Hybrid(StaticModel(cbf_scores), StaticModel(cf_scores), 1).score(
            np.array(['u']), np.array(['a', 'b']))
        self.assertEqual(got[0, 0], 1.0)
        self.assertEqual(got[0, 1], 0.9)

    def test_invalid_inputs(self):
        r = row('a', 'p', 0)
        for cbf, cf, alpha in [([r, r], [], .5),
                               ([r], [row('a', 'q', 3)], .5),
                               ([r], [], 2),
                               ([row('a', 'p', float('nan'))], [], .5)]:
            with self.assertRaises(ValueError):
                combine_scores(cbf, cf, alpha)

    def test_couple_methods_and_seen_union(self):
        rows = [row('a', 'p', 1), row('b', 'p', .2),
                row('a', 'q', .55), row('b', 'q', .55)]
        self.assertEqual(couple_recommendations(rows, 'a', 'b', 'mean')[0]['place_id'], 'p')
        self.assertEqual(couple_recommendations(rows, 'a', 'b', 'min')[0]['place_id'], 'q')
        self.assertEqual(couple_recommendations(rows, 'a', 'b', excluded={'p'})[0]['place_id'], 'q')
        rows[0]['year'] = '2023'
        with self.assertRaises(ValueError):
            couple_recommendations(rows, 'a', 'b')


if __name__ == '__main__':
    unittest.main()
