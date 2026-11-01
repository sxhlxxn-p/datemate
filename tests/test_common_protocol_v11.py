import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import eval_protocol as ep
from cbf import CBF


class Model:
    def __init__(self, scores):
        self.scores = np.array(scores, dtype=float)

    def score(self, users, cand):
        return self.scores

    def predict(self, users, places):
        return None


def data():
    # u1: warm positive; u2: cold positive. Both have the same single seen item.
    train = pd.DataFrame([('u1', 'p0', 4.), ('u2', 'p0', 4.), ('helper', 'p1', 4.)],
                         columns=['user_id', 'place_id', 'rating'])
    ev = pd.DataFrame([('u1', 'p1', 5.), ('u2', 'p2', 5.)], columns=train.columns)
    return dict(year='2022', train=train, validation=ev, test=ev.copy(),
                cand=np.array(['p0', 'p1']), global_mean=4., candidate_mode='warm')


class CommonProtocolTests(unittest.TestCase):
    def test_unsupported_relevant_item_is_not_a_hit_or_removed_user(self):
        d = data()
        m, p = ep.evaluate(d, Model([[0, -np.inf], [0, -np.inf]]), 'validation')
        self.assertEqual(m['ranked_users'], 2)
        self.assertEqual(m['warm_ranked_users'], 1)
        self.assertEqual(m['users_without_scores'], 2)
        self.assertEqual(m['recall_at_10'], 0.)
        self.assertEqual(m['recall_at_10_warm'], 0.)
        self.assertEqual(m['ndcg_at_10'], 0.)
        self.assertEqual(m['recommendable_user_ratio'], 0.)
        self.assertTrue(p.empty)

    def test_warm_and_all_candidates_have_distinct_meanings(self):
        d = data()
        m, _ = ep.evaluate(d, Model([[0, 1], [0, 1]]), 'validation')
        self.assertEqual(m['recall_at_10'], .5)  # cold target cannot be recommended
        self.assertEqual(m['recall_at_10_warm'], 1.)
        self.assertEqual(m['relevant_cold'], 1)
        d.update(cand=np.array(['p0', 'p1', 'p2']), candidate_mode='all')
        m, _ = ep.evaluate(d, Model([[0, 1, 0], [0, 0, 1]]), 'validation')
        self.assertEqual(m['recall_at_10'], 1.)
        self.assertEqual(m['relevant_warm'], 1)  # cand membership must not redefine warm
        self.assertEqual(m['relevant_cold'], 1)
        self.assertEqual(m['recommendable_user_ratio'], 0.)  # fewer than ten valid recs

    def test_no_positive_cohort_is_null_and_json_is_strict(self):
        d = data()
        d['validation'] = d['validation'].assign(rating=3.)
        m, _ = ep.evaluate(d, Model([[0, 1], [0, 1]]), 'validation')
        self.assertIsNone(m['recall_at_10'])
        self.assertIsNone(m['recall_at_10_warm'])
        self.assertIsNone(m['recommendable_user_ratio'])
        json.dumps(m, allow_nan=False)

    def test_hand_calculated_macro_ndcg_and_tie_order(self):
        d = data()
        d['validation'] = pd.DataFrame([('u1', 'p2', 5.), ('u1', 'p3', 4.), ('u2', 'p1', 5.)],
                                       columns=d['train'].columns)
        d.update(cand=np.array(['p0', 'p1', 'p2', 'p3']), candidate_mode='all')
        scores = [[10, 3, 2, 1], [10, 3, 2, 1]]
        m, p = ep.evaluate(d, Model(scores), 'validation')
        expected_u1 = (1 / np.log2(3) + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
        self.assertAlmostEqual(m['ndcg_at_10'], (expected_u1 + 1) / 2)
        self.assertEqual(m['recall_at_10'], 1.)
        self.assertNotIn('p0', p.place_id.tolist())
        _, p = ep.evaluate(d, Model(np.zeros((2, 4))), 'validation')
        self.assertEqual(p[p.user_id == 'u1'].place_id.tolist(), ['p1', 'p2', 'p3'])

    def test_idcg_uses_all_relevant_targets_even_when_candidates_are_few(self):
        d = data()
        d['validation'] = pd.DataFrame([('u1', p, 5.) for p in ('p1', 'cold2', 'cold3', 'cold4', 'cold5')],
                                       columns=d['train'].columns)
        m, _ = ep.evaluate(d, Model([[0, 1]]), 'validation')
        self.assertEqual(m['recall_at_10'], .2)
        self.assertAlmostEqual(m['ndcg_at_10'], 1 / ep.DISC[:5].sum())

    def test_load_all_uses_fixed_candidate_flag_and_means_after_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame([
                ('2022', 'u', 'p0', 3, 'train'), ('2022', 'u', 'p0', 5, 'train'),
                ('2022', 'u', 'p1', 5, 'validation'), ('2022', 'u', 'p2', 4, 'test')],
                columns=['year', 'user_id', 'place_id', 'rating', 'known_user_split']).to_csv(root/'interactions.csv', index=False)
            pd.DataFrame([('p0', 1), ('p1', 1), ('p2', 1), ('excluded', 0)],
                         columns=['place_id', 'is_model_candidate']).to_csv(root/'places.csv', index=False)
            d = ep.load_data('2022', root, 'all')
            self.assertEqual(d['train'].rating.tolist(), [4.])
            self.assertEqual(d['train_cand'].tolist(), ['p0'])
            self.assertEqual(d['cand'].tolist(), ['p0', 'p1', 'p2'])
            self.assertNotEqual(d['protocol'], ep.load_data('2022', root)['protocol'])

    def test_cbf_fits_train_vocabulary_and_scores_unseen_catalog(self):
        d = data()
        ids = ['p0', 'p1', 'p2']
        tables = {k: pd.Series(values, index=ids) for k, values in {
            'name': ['cafe park', 'museum park', 'coldonly cafe'],
            'vtype': ['1', '1', '2'], 'sgg': ['', '', '']}.items()}
        # Candidate-mode must not affect learned vector dimension or train profile.
        a = CBF(w_type=0).fit(d, tables)
        all_data = dict(d, cand=np.array(ids), candidate_mode='all')
        b = CBF(w_type=0).fit(all_data, tables)
        self.assertEqual(a.X.shape, b.X.shape)
        sa = a.score(np.array(['u1', 'u2']), np.array(ids))
        np.testing.assert_allclose(sa, b.score(np.array(['u1', 'u2']), np.array(ids)))
        self.assertGreater(sa[0, 2], 0)
        np.testing.assert_allclose(sa[:, [2, 1]], a.score(np.array(['u1', 'u2']), np.array(['p2', 'p1'])))

    def test_result_modes_have_separate_paths_and_reproducibility_hashes(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for file in ('interactions.csv', 'places.csv', 'users.csv'):
                (root / file).write_text('fictional_fixture\n')
            paths = []
            for mode in ('warm', 'all'):
                d = dict(data(), candidate_mode=mode, protocol=f'v1_1/known_user_2022/{mode}')
                m, pred = ep.evaluate(d, Model([[0, 1], [0, 1]]), 'validation')
                with patch.object(ep, 'ROOT', root):
                    out = ep.save_results(d, 'fixture', 'validation', m, pred, root, {'top_n': 10}, 0.)
                saved = json.loads((out / 'metrics.json').read_text())
                self.assertEqual(saved['meta']['candidate_mode'], mode)
                self.assertEqual(saved['meta']['predictions_top'], 10)
                self.assertEqual(len(saved['meta']['warm_positive_users_sha256']), 64)
                paths.append(out)
            self.assertNotEqual(*paths)


if __name__ == '__main__':
    unittest.main()
