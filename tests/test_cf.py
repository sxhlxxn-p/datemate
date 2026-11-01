import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cf import ItemKNN, tune_k
import eval_protocol as ep


def fixture():
    train = pd.DataFrame([('a','p0',4.), ('a','p1',5.), ('b','p1',4.),
                          ('b','p2',5.), ('c','p0',4.), ('c','p2',4.)],
                         columns=['user_id','place_id','rating'])
    ev = pd.DataFrame([('a','p2',5.), ('b','cold',5.)], columns=train.columns)
    return dict(year='2022', train=train, validation=ev, test=ev.copy(),
                cand=np.array(['p0','p1','p2']), global_mean=train.rating.mean())


class CFTests(unittest.TestCase):
    def test_score_matches_predict_and_preserves_requested_order(self):
        d = fixture()
        model = ItemKNN(2).fit(d)
        users = np.array(['b','a','unknown'])
        cand = np.array(['cold','p2','p0'])
        scores = model.score(users,cand)
        self.assertEqual(scores.shape,(3,3))
        self.assertTrue(np.isneginf(scores[:,0]).all())
        self.assertTrue(np.isneginf(scores[2]).all())
        for i,u in enumerate(users):
            for j,p in enumerate(cand):
                pred = model.predict([u],[p])[0]
                if np.isfinite(scores[i,j]):
                    self.assertAlmostEqual(scores[i,j],pred)
                else:
                    self.assertTrue(np.isnan(pred))

    def test_warm_all_share_users_and_train_model(self):
        d = fixture()
        model = ItemKNN(2).fit(d)
        original = model.R.copy()
        for mode,cand in [('warm',['p0','p1','p2']), ('all',['cold','p0','p1','p2'])]:
            m,p = ep.evaluate(dict(d,candidate_mode=mode,cand=np.array(cand)),model,'test')
            self.assertEqual(m['ranked_users'],2)
            self.assertEqual(m['warm_ranked_users'],1)
            self.assertEqual(m['relevant_cold'],1)
            self.assertNotIn('cold',p.place_id.tolist())
        self.assertEqual((model.R != original).nnz,0)

    def test_no_neighbors_is_unsupported(self):
        d = fixture()
        d['train'] = pd.DataFrame([('a','p0',4.)],columns=d['train'].columns)
        model=ItemKNN(20).fit(d)
        self.assertTrue(np.isneginf(model.score(['a'],['p0','cold'])).all())
        self.assertTrue(np.isnan(model.predict(['a'],['p0'])).all())
        with self.assertRaises(ValueError):
            ItemKNN(0)

    def test_identical_items_keep_exactly_k_nonself_neighbors_in_id_order(self):
        train = pd.DataFrame([('u',f'p{i}',4.) for i in range(8)],
                             columns=['user_id','place_id','rating'])
        d = dict(train=train,cand=np.array(sorted(train.place_id.unique())))
        model = ItemKNN(1).fit(d)
        self.assertTrue((np.diff(model.S.indptr) == 1).all())
        self.assertTrue((model.S.diagonal() == 0).all())
        for i in range(8):
            expected = 1 if i == 0 else 0
            self.assertEqual(model.S.getrow(i).indices.tolist(),[expected])
        shuffled = dict(d,train=train.sample(frac=1,random_state=7))
        other = ItemKNN(1).fit(shuffled)
        np.testing.assert_allclose(model.S.toarray(),other.S.toarray())

    def test_tuning_uses_warm_ndcg_and_grid_order_for_ties(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            def fake_evaluate(d,root,k,*args,**kwargs):
                metrics=dict(recall_at_10=0., ndcg_at_10=1. if k==10 else 0.,
                             recall_at_10_warm=0.,ndcg_at_10_warm=.5 if k in (20,40) else .1,
                             recommendable_user_ratio=0., item_coverage=0.,mae=None,rmse=None,
                             mae_supported=None,rmse_supported=None,rating_supported_pairs=0)
                return metrics,0.
            with patch('cf.ROOT',Path(tmp)), patch('cf.evaluate_one',side_effect=fake_evaluate):
                self.assertEqual(tune_k(fixture(),Path(tmp),'2022',[10,40,20]),40)


if __name__ == '__main__':
    unittest.main()
