import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import eval_protocol as ep
import run_mf
from mf import MatrixFactorization


def frame(rows):
    return pd.DataFrame(rows,columns=['user_id','place_id','rating'])


def data():
    t=frame([('u','a',5),('u','b',2),('v','b',5),('v','c',2),('w','a',4),('w','c',3)])
    return {'train':t,'validation':frame([('u','c',4)]),'test':frame([('v','a',4)])}


class MFTests(unittest.TestCase):
    def test_observed_objective_decreases_and_reproducible(self):
        d=data(); a=MatrixFactorization(2,.1,10).fit(d)
        b=MatrixFactorization(2,.1,10).fit(d)
        np.testing.assert_allclose(a.P,b.P,atol=0,rtol=0)
        self.assertTrue(all(y<=x+1e-9 for x,y in zip(a.loss_history,a.loss_history[1:])))
        self.assertEqual(a.mu,d['train'].rating.mean())
        changed=dict(d,test=frame([('v','a',1)]),validation=frame([('u','c',1)]))
        c=MatrixFactorization(2,.1,10).fit(changed)
        np.testing.assert_array_equal(a.Q,c.Q)

    def test_order_predict_alignment_and_cold_behavior(self):
        m=MatrixFactorization(2,1,5).fit(data())
        s=m.score(['v','u','new'],['c','a','cold'])
        self.assertEqual(s.shape,(3,3))
        np.testing.assert_allclose(s[[0,1],[0,1]],m.predict(['v','u'],['c','a']))
        self.assertTrue(np.isneginf(s[2]).all())
        self.assertTrue(np.isneginf(s[:,2]).all())
        self.assertTrue(np.isnan(m.predict(['u','new'],['cold','a'])).all())

    def test_fold_in_solves_ridge_from_support_and_freezes_items(self):
        m=MatrixFactorization(2,1,5).fit(data()); q=m.Q.copy(); p=m.P.copy(); bias=m.bi.copy()
        support=frame([('new','a',5),('new','b',2),('no_known','cold',4)])
        n=m.with_history(support)
        ii=m.item_idx.loc[['a','b']].to_numpy(int)
        x=np.column_stack([np.ones(2),m.Q[ii]])
        expected=np.linalg.solve(x.T@x+np.eye(3), x.T@(np.array([5,2])-m.mu-m.bi[ii]))
        np.testing.assert_allclose(np.r_[n.bu[0],n.P[0]],expected)
        np.testing.assert_array_equal(m.Q,q); np.testing.assert_array_equal(m.P,p)
        np.testing.assert_array_equal(n.bi,bias)
        self.assertTrue(np.isneginf(n.score(['no_known'],['a'])).all())
        self.assertTrue(np.isfinite(n.score(['new'],['c'])).all())
        d=dict(data(),test=frame([('new','c',5)]))
        n2=m.with_history(support)
        np.testing.assert_array_equal(n.P,n2.P)

    def test_bias_only_and_bad_inputs(self):
        m=MatrixFactorization(0,1,5).fit(data())
        self.assertEqual(m.P.shape,(3,0))
        self.assertTrue(np.isfinite(m.predict(['u'],['c'])).all())
        for kwargs in [dict(factors=-1),dict(reg=0),dict(epochs=0)]:
            with self.assertRaises(ValueError): MatrixFactorization(**kwargs)
        for t in [frame([]),frame([('u','a',0)]),frame([('u','a',np.nan)]),frame([('u','a',5),('u','a',4)])]:
            with self.assertRaises(ValueError): MatrixFactorization().fit({'train':t})

    def test_common_evaluation_masks_support_and_retains_train_warmth(self):
        m=MatrixFactorization(2,1,5).fit(data()).with_history(frame([('new','a',5)]))
        d=dict(data(),history=frame([('new','a',5)]),external_test=frame([('new','b',5)]),
               cand=np.array(['a','b','c','cold']),global_mean=3.5,candidate_mode='all',year='2023')
        metrics,preds=ep.evaluate(d,m,'external_test',top_n=10)
        self.assertNotIn('a',set(preds.place_id));self.assertNotIn('cold',set(preds.place_id))
        self.assertEqual(metrics['relevant_warm'],1)
        self.assertEqual(metrics['rating_supported_pairs'],1)

    def test_tune_evaluate_roundtrip_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); inp=path/'data';inp.mkdir()
            d=data(); rows=[]
            for split in ('train','validation','test'):
                g=d[split].assign(year='2022',known_user_split=split);rows.append(g)
            pd.concat(rows).to_csv(inp/'interactions.csv',index=False)
            pd.DataFrame({'place_id':['a','b','c'],'is_model_candidate':['1']*3}).to_csv(inp/'places.csv',index=False)
            pd.DataFrame({'user_id':['u','v','w']}).to_csv(inp/'users.csv',index=False)
            config=dict(run_mf.CONFIG,factors_grid=[1],regularization_grid=[1.],epochs=2)
            with patch.object(run_mf,'CONFIG',config),patch.object(ep,'ROOT',path):
                run_mf.tune(inp,'internal',path/'out')
                result=run_mf.evaluate(inp,'internal',path/'out')
                self.assertTrue(result['matching_input_manifests'])
                frame([('new','a',5),('no_known','cold',4)]).to_csv(inp/'support.csv',index=False)
                frame([('new','b',5),('no_known','c',5)]).to_csv(inp/'external_targets.csv',index=False)
                run_mf.tune(inp,'external',path/'external')
                ext=run_mf.evaluate(inp,'external',path/'external')
                self.assertTrue(ext['matching_input_manifests'])
                external_runs=[r for r in ext['runs'] if r['meta']['split']=='external_test']
                pure=next(r for r in external_runs if r['meta']['model']=='mf')
                fallback=next(r for r in external_runs if r['meta']['model']=='mf_or_popularity')
                self.assertEqual(pure['metrics']['users_without_scores'],1)
                self.assertEqual(fallback['metrics']['users_without_scores'],0)
                (inp/'users.csv').write_text('user_id\nmodified\n')
                with self.assertRaisesRegex(ValueError,'changed'):run_mf.evaluate(inp,'internal',path/'out')


if __name__=='__main__':unittest.main()
