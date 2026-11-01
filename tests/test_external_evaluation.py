import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import external_evaluation as ex
import eval_protocol as ep
from cbf import CBF
from cf import ItemKNN
from make_synthetic_data import generate


def history_fixture():
    train=pd.DataFrame([('a','p0',4.),('a','p1',5.),('b','p1',4.),
                        ('b','p2',5.),('c','p0',4.),('c','p2',4.)],columns=ex.PAIR_COLUMNS)
    d=dict(train=train,cand=np.array(['p0','p1','p2']),year='2022',global_mean=train.rating.mean())
    tables={k:pd.Series(v,index=['p0','p1','p2']) for k,v in {
        'name':['cafe park','museum park','museum cafe'], 'vtype':['1','1','2'],
        'sgg':['seoul a','seoul a','seoul b'], 'sido':['seoul']*3}.items()}
    return d,tables


class ExternalTests(unittest.TestCase):
    def test_day_boundary_repeats_and_future_rating_do_not_leak_into_support(self):
        rows=pd.DataFrame([
            ('u','p0',4.,'2023-01-01','i0'),('u','p1',5.,'2023-01-01','i1'),
            ('u','p2',3.,'2023-01-01','i2'),('u','p3',4.,'2023-01-01','i3'),
            ('u','p0',1.,'2023-01-02','i4'),('u','q',5.,'2023-01-02','i5')],
            columns=['user_id','place_id','rating','visit_date','interaction_id'])
        support,query,_,audit,_=ex.split_support_query(rows)
        self.assertEqual(set(support.place_id),{'p0','p1','p2','p3'})
        self.assertEqual(support.set_index('place_id').loc['p0','rating'],4.)
        self.assertEqual(query.place_id.tolist(),['q'])
        self.assertEqual(audit['later_repeats_of_seen_venues_ignored'],1)
        changed=rows.copy();changed.loc[changed.visit_date=='2023-01-02','rating']=2.
        again=ex.split_support_query(changed)[0]
        pd.testing.assert_frame_equal(support,again)

    def test_no_future_query_is_rejected_and_bad_dates_counted(self):
        rows=pd.DataFrame([('u',f'p{i}',4.,'2023-01-01',str(i)) for i in range(4)],
            columns=['user_id','place_id','rating','visit_date','interaction_id'])
        with self.assertRaisesRegex(ValueError,'No users'):
            ex.split_support_query(rows)

    def test_history_cf_matches_original_and_handles_new_user(self):
        d,_=history_fixture();base=ItemKNN(2).fit(d)
        users=np.array(['a','b']);cand=np.array(['p2','cold','p0'])
        wrapper=ex.HistoryCF(base,d['train'])
        np.testing.assert_allclose(wrapper.score(users,cand),base.score(users,cand))
        np.testing.assert_allclose(wrapper.predict(users,np.array(['p2','p0'])),base.predict(users,np.array(['p2','p0'])),equal_nan=True)
        new=d['train'][d['train'].user_id=='a'].assign(user_id='2023:new')
        adapter=ex.HistoryCF(base,new)
        np.testing.assert_allclose(adapter.score(np.array(['2023:new']),cand),base.score(np.array(['a']),cand))
        self.assertTrue(np.isneginf(adapter.score(np.array(['missing']),cand)).all())

    def test_history_cbf_matches_original_and_new_user_profile(self):
        d,tables=history_fixture();base=CBF(w_type=.5,w_region=.5).fit(d,tables)
        users=np.array(['a','b']);cand=np.array(['p2','p0','p1'])
        wrapper=ex.HistoryCBF(base,d['train'])
        np.testing.assert_allclose(wrapper.score(users,cand),base.score(users,cand),atol=1e-12)
        np.testing.assert_allclose(wrapper.predict(users,np.array(['p2','p0'])),base.predict(users,np.array(['p2','p0'])),atol=1e-12)
        new=d['train'][d['train'].user_id=='a'].assign(user_id='2023:new')
        np.testing.assert_allclose(ex.HistoryCBF(base,new).score(np.array(['2023:new']),cand),base.score(np.array(['a']),cand),atol=1e-12)

    def test_evaluator_excludes_seed_without_redefining_training_warmth(self):
        d,_=history_fixture()
        d.update(cand=np.array(['cold','p0','p1','p2']),protocol_version='external_v2',
                 history=pd.DataFrame([('new','cold',4.)],columns=ex.PAIR_COLUMNS),
                 external_test=pd.DataFrame([('new','p2',5.)],columns=ex.PAIR_COLUMNS))
        class Model:
            def score(self,users,cand): return np.array([[100.,0.,0.,1.]])
            def predict(self,users,places): return None
        m,p=ep.evaluate(d,Model(),'external_test')
        self.assertNotIn('cold',p.place_id.tolist())
        self.assertEqual(m['relevant_warm'],1)
        self.assertEqual(m['protocol_version'],'external_v2')

    def test_snapshot_2022_is_invariant_to_2023_target_name_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)/'raw';generate(raw)
            first=Path(tmp)/'first';second=Path(tmp)/'second'
            with patch('builtins.print'):
                ex.prepare(raw,first)
            for f in raw.glob('synthetic_2023/*/tn_visit_area_info*.csv'):
                rows=pd.read_csv(f,dtype=str)
                rows['VISIT_AREA_NM']='changed future target name'
                rows['DGSTFN']='1'
                rows.to_csv(f,index=False)
            with patch('builtins.print'):
                ex.prepare(raw,second)
            self.assertEqual((first/'places.csv').read_bytes(),(second/'places.csv').read_bytes())
            for path in [first,second]:
                quality=__import__('json').loads((path/'quality_report.json').read_text())
                self.assertFalse(quality['poi_master_used'])
                self.assertEqual(quality['catalog_years'],['2022'])

    def test_seed_features_are_per_user_and_do_not_change_fitted_catalog_vectors(self):
        d,tables=history_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pd.DataFrame([('new1','new-place','cafe','서울 구1','','1'),
                          ('new2','new-place','museum','서울 구2','','2')],
                         columns=['user_id','place_id','place_name','road_address','lot_address','visit_type_code']).to_csv(root/'support_features.csv',index=False)
            extended,keys=ex.support_feature_tables(root,tables)
            self.assertNotEqual(keys['new1','new-place'],keys['new2','new-place'])
            old=CBF(w_type=.5,w_region=.5).fit(d,tables)
            new=CBF(w_type=.5,w_region=.5).fit(d,extended)
            np.testing.assert_allclose(old.X.toarray(),new.X[:3].toarray(),atol=1e-12)
            seeds=pd.DataFrame([('new1','new-place',5.),('new2','new-place',5.)],columns=ex.PAIR_COLUMNS)
            score=ex.HistoryCBF(new,seeds,keys).score(np.array(['new1','new2']),d['cand'])
            self.assertFalse(np.allclose(score[0],score[1]))

    def test_no_cf_candidate_routes_to_popularity_without_weighted_fusion(self):
        d,_=history_fixture();base=ItemKNN(2).fit(d)
        seeds=pd.DataFrame([('new','outside',5.)],columns=ex.PAIR_COLUMNS)
        cf=ex.HistoryCF(base,seeds);pop=ep.Baseline('popularity').fit(d)
        routed=ex.AvailabilityFallbackCF(cf,pop)
        users=np.array(['new'])
        np.testing.assert_allclose(routed.score(users,d['cand']),pop.score(users,d['cand']))
        self.assertIsNone(routed.predict(users,np.array(['p0'])))

    def test_fictional_prepare_tune_evaluate_and_frozen_data_guard(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';generate(raw)
            # Helpers keep held-out p5/p6 warm in the 2022 fictional training catalog.
            for year in ('2022','2023'):
                base=raw/f'synthetic_{year}'/'TL_csv'
                for pattern, record in [
                    ('tn_traveller_master*',{'TRAVELER_ID':'SYNTH_HELPER','GENDER':'','AGE_GRP':''}),
                    ('tn_travel_info*',{'TRAVEL_ID':'SYNTH_HELP_TRIP','TRAVELER_ID':'SYNTH_HELPER'})]:
                    path=next(base.glob(pattern));table=pd.read_csv(path,dtype=str)
                    pd.concat([table,pd.DataFrame([record])],ignore_index=True).to_csv(path,index=False)
                path=next(base.glob('tn_visit_area_info*'));table=pd.read_csv(path,dtype=str)
                extra=[]
                for i in (5,6):
                    r=table.iloc[0].to_dict();r.update(TRAVEL_ID='SYNTH_HELP_TRIP',VISIT_AREA_ID=f'SYNTH_H{i}',
                        POI_ID=f'SYNTH_P{i}',VISIT_AREA_NM=f'가상장소{i}',ROAD_NM_ADDR=f'서울 가상주소 {i}',DGSTFN='5')
                    extra.append(r)
                pd.concat([table,pd.DataFrame(extra)],ignore_index=True).to_csv(path,index=False)
            data=root/'data';out=root/'run'
            with patch('builtins.print'):
                ex.prepare(raw,data)
                frozen=ex.tune(data,out)
                with patch.object(ep,'ROOT',root):
                    results=ex.evaluate_all(data,out)
            self.assertTrue(results)
            self.assertEqual(frozen['protocol']['c_part'],'excluded')
            self.assertTrue(all(x['metrics']['protocol_version']=='external_v2' for x in results))
            self.assertTrue(any(x['meta']['year']=='2023' for x in results))
            with (data/'external_targets.csv').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'changed after tuning'):
                ex.evaluate_all(data,out)


if __name__=='__main__': unittest.main()
