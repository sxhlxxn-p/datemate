import unittest
from datemate.hybrid import combine_scores, couple_recommendations


def row(user, place, score):
    return dict(user_id=user, place_id=place, score=score, year='2022', split='validation')


class HybridTests(unittest.TestCase):
    def test_fallback_and_endpoints(self):
        cbf = [row('a', 'p', 0), row('a', 'q', 1)]
        cf = [row('a', 'p', 5)]
        scores = combine_scores(cbf, cf, 1)
        self.assertEqual([r['score'] for r in scores], [1, 1])
        self.assertEqual(scores[1]['score_source'], 'cbf_fallback')
        self.assertEqual(combine_scores(cbf, cf, 0)[0]['score'], .5)

    def test_invalid_inputs(self):
        r = row('a', 'p', 0)
        for cbf, cf, alpha in [([r, r], [], .5), ([r], [row('a','q',3)], .5),
                               ([r], [], 2), ([row('a','p',float('nan'))], [], .5)]:
            with self.assertRaises(ValueError):
                combine_scores(cbf, cf, alpha)

    def test_couple_methods_and_seen_union(self):
        rows = [row('a','p',1), row('b','p',.2), row('a','q',.55), row('b','q',.55)]
        self.assertEqual(couple_recommendations(rows,'a','b','mean')[0]['place_id'], 'p')
        self.assertEqual(couple_recommendations(rows,'a','b','min')[0]['place_id'], 'q')
        self.assertEqual(couple_recommendations(rows,'a','b',excluded={'p'})[0]['place_id'], 'q')
        rows[0]['year'] = '2023'
        with self.assertRaises(ValueError):
            couple_recommendations(rows,'a','b')


if __name__ == '__main__':
    unittest.main()
