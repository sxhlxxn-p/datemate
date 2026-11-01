"""2022-only catalog and 2023 support/query evaluation; preserves all v1.1 outputs.

Stages: prepare -> tune (2022 validation only) -> evaluate (frozen settings).
Raw source files are read-only. C/Hybrid and couple recommendations are excluded.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

import build_common_dataset as builder
import cbf
from cf import ItemKNN
import eval_protocol as ep

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs/external_evaluation_v2.json'
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
if CONFIG['k'] != ep.K or CONFIG['relevance_rating_gte'] != ep.REL:
    raise ValueError('External K/relevance must match the common metric implementation')
PAIR_COLUMNS = ['user_id', 'place_id', 'rating']


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def pair_mean(rows):
    return rows.groupby(['user_id', 'place_id'], as_index=False).rating.mean()


def frozen_aliases(visits):
    """Only 2022 rows can establish aliases for the 2022 identity space."""
    addr, coord = defaultdict(set), defaultdict(set)
    for r in visits:
        pid = r.get('place_id', '')
        if not pid:
            continue
        names = {builder.norm_text(r.get(k)) for k in ('VISIT_AREA_NM', 'POI_NM')} - {''}
        addresses = {builder.norm_text(r.get(k)) for k in ('ROAD_NM_ADDR', 'LOTNO_ADDR')} - {''}
        xy = builder.coord_key(r.get('X_COORD'), r.get('Y_COORD'))
        for n in names:
            for a in addresses:
                addr[n, a].add(pid)
            if xy:
                coord[n, xy].add(pid)
    return addr, coord


def resolve_external(row, aliases):
    poi = builder.clean(row.get('POI_ID'))
    if poi:
        return 'poi:' + poi
    names = {builder.norm_text(row.get(k)) for k in ('VISIT_AREA_NM', 'POI_NM')} - {''}
    addresses = {builder.norm_text(row.get(k)) for k in ('ROAD_NM_ADDR', 'LOTNO_ADDR')} - {''}
    matches = set().union(*(aliases[0].get((n, a), set()) for n in names for a in addresses))
    if len(matches) == 1:
        return next(iter(matches))
    xy = builder.coord_key(row.get('X_COORD'), row.get('Y_COORD'))
    matches = set().union(*(aliases[1].get((n, xy), set()) for n in names)) if xy else set()
    if len(matches) == 1:
        return next(iter(matches))
    name = builder.norm_text(row.get('VISIT_AREA_NM') or row.get('POI_NM'))
    for field, prefix in [('ROAD_NM_ADDR', 'addr'), ('LOTNO_ADDR', 'lot')]:
        address = builder.norm_text(row.get(field))
        if name and address:
            return builder.stable_id(prefix, name, address)
    return builder.stable_id('coord', name, xy) if name and xy else ''


def clean_external(visits, aliases, catalog):
    lookup = {r['place_id']: r for r in catalog}
    out, drops = [], Counter()
    for source in visits:
        r = dict(source)
        rating = builder.to_int(r.get('DGSTFN'))
        reason = None
        if not r.get('user_id') or r.get('profile_exists') != '1':
            reason = 'missing_user_or_profile'
        elif rating not in (1, 2, 3, 4, 5):
            reason = 'missing_or_invalid_rating'
        elif r.get('VISIT_AREA_TYPE_CD') not in builder.RECOMMENDABLE_TYPES:
            reason = 'excluded_visit_type'
        if reason:
            drops[reason] += 1
            continue
        pid = resolve_external(r, aliases)
        if not pid:
            drops['unresolved_place'] += 1
            continue
        old = lookup.get(pid, {})
        # Old catalog metadata can fill missing fields; no 2023 target contributes to the catalog.
        for raw, field in [('ROAD_NM_ADDR','road_address'), ('LOTNO_ADDR','lot_address'), ('SGG_CD','sgg_code')]:
            if not builder.clean(r.get(raw)):
                r[raw] = old.get(field, '')
        if not builder.is_capital_area(r):
            drops['outside_or_unknown_capital_area'] += 1
            continue
        out.append(dict(
            interaction_id=r['interaction_id'], user_id=r['user_id'], place_id=pid,
            year='2023', source_split=r['source_split'], travel_id=r['TRAVEL_ID'],
            visit_area_id=r['VISIT_AREA_ID'], visit_date=r.get('VISIT_START_YMD',''),
            visit_order=r.get('VISIT_ORDER',''), rating=rating,
            known_user_split='external_unassigned',
            place_name=r.get('VISIT_AREA_NM') or r.get('POI_NM',''),
            road_address=r.get('ROAD_NM_ADDR',''), lot_address=r.get('LOTNO_ADDR',''),
            visit_type_code=r.get('VISIT_AREA_TYPE_CD','')))
    if not out:
        raise ValueError('No eligible external records')
    return pd.DataFrame(out), dict(drops)


def split_support_query(rows, min_unique=3):
    """Cut at the end of the third distinct venue's day, before reading any target rating.

    All visits through that day are support; subsequent visits to support venues
    are ignored. Targets are later-day, unseen venues. Aggregation happens last.
    """
    if min_unique < 1:
        raise ValueError('min_unique must be positive')
    g = rows.copy()
    g['day'] = pd.to_datetime(g.visit_date, errors='coerce', format='mixed').dt.normalize()
    audit = Counter(invalid_date_rows=int(g.day.isna().sum()))
    g = g[g.day.notna()].sort_values(['user_id', 'day', 'interaction_id'], kind='stable')
    supports, targets, features = [], [], []
    cutoffs = {}
    for user, u in g.groupby('user_id', sort=True):
        first = u.drop_duplicates('place_id')
        if len(first) < min_unique:
            audit['users_below_support_minimum'] += 1
            continue
        cutoff = first.iloc[min_unique - 1].day
        initial = u[u.day <= cutoff]
        seen = set(initial.place_id)
        query = u[(u.day > cutoff) & ~u.place_id.isin(seen)]
        if query.empty:
            audit['users_without_later_unseen_targets'] += 1
            continue
        supports.append(initial)
        targets.append(query)
        features.append(initial.drop_duplicates('place_id'))
        cutoffs[str(user)] = cutoff.strftime('%Y-%m-%d')
        audit['later_repeats_of_seen_venues_ignored'] += int(((u.day > cutoff) & u.place_id.isin(seen)).sum())
    if not supports:
        raise ValueError('No users have both initial support and later unseen targets')
    support_rows = pd.concat(supports, ignore_index=True)
    query_rows = pd.concat(targets, ignore_index=True)
    support, query = pair_mean(support_rows), pair_mean(query_rows)
    if set(map(tuple, support[['user_id','place_id']].to_numpy())) & set(map(tuple, query[['user_id','place_id']].to_numpy())):
        raise ValueError('Support/query user-place leakage')
    last_support = support_rows.groupby('user_id').day.max()
    first_query = query_rows.groupby('user_id').day.min()
    if not (last_support < first_query).all():
        raise ValueError('Support/query dates are not strictly ordered')
    audit.update(eligible_users=int(support.user_id.nunique()), support_pairs=len(support),
                 query_pairs=len(query), support_rows=len(support_rows), query_rows=len(query_rows))
    return support, query, pd.concat(features, ignore_index=True), dict(audit), cutoffs


def prepare(raw_dir, output):
    if output.resolve() == raw_dir.resolve() or raw_dir.resolve() in output.resolve().parents:
        raise ValueError('Output must be outside raw source data')
    audit, visits = builder.load_sources(raw_dir)
    if audit['duplicate_interaction_ids']:
        raise ValueError('Source interaction IDs are not unique')
    profiles = audit.pop('_users')
    v22 = [r for r in visits if r['year'] == '2022']
    v23 = [r for r in visits if r['year'] == '2023']
    date_years={}
    for year, records in [('2022',v22),('2023',v23)]:
        dates=pd.to_datetime(pd.Series([r.get('VISIT_START_YMD','') for r in records]),errors='coerce',format='mixed')
        valid=dates.dropna()
        if not (valid.dt.year==int(year)).all():
            raise ValueError(f"Source label {year} contains visits in a different calendar year")
        date_years[year]=dict(valid_rows=int(dates.notna().sum()),invalid_rows=int(dates.isna().sum()),
                              min=str(valid.min().date()),max=str(valid.max().date()))
    # No POI master: its historical availability is not known.
    builder.resolve_places(v22, {})
    catalog = builder.build_places(v22, {})
    interactions22, prep22 = builder.prepare_interactions(v22, catalog)
    external, drops = clean_external(v23, frozen_aliases(v22), catalog)
    support, query, features, split_audit, cutoffs = split_support_query(
        external, CONFIG['support_min_unique_places'])
    output.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.DataFrame(interactions22), external], ignore_index=True).to_csv(output/'interactions.csv', index=False)
    pd.DataFrame(catalog).to_csv(output/'places.csv', index=False)
    users = builder.build_users(profiles, interactions22 + external.to_dict('records'))
    pd.DataFrame(users).to_csv(output/'users.csv', index=False)
    for name, table in [('support',support),('external_targets',query),('support_features',features)]:
        table.drop(columns='day', errors='ignore').to_csv(output/(name+'.csv'), index=False)
    write_json(output/'support_cutoffs.json', cutoffs)
    quality = dict(protocol=CONFIG, source=audit, preprocessing_2022=prep22,
                   interactions_2022=len(interactions22), eligible_records_2023=len(external),
                   catalog_rows=len(catalog), catalog_candidates=sum(p['is_model_candidate']=='1' for p in catalog),
                   external_drop_reasons=drops, external_split=split_audit,
                   poi_master_used=False, catalog_years=['2022'], source_visit_dates=date_years)
    write_json(output/'quality_report.json', quality)
    print(json.dumps(quality, ensure_ascii=False, indent=2))
    return quality


def load_protocol(data_dir, mode='warm_2022', external=False):
    d = ep.load_data('2022', data_dir, 'warm')
    if mode not in CONFIG['external_candidate_modes']:
        raise ValueError('Unknown external candidate mode')
    if mode == 'catalog_2022':
        places = pd.read_csv(data_dir/'places.csv', dtype=str)
        d['cand'] = np.array(sorted(places.loc[places.is_model_candidate == '1','place_id']))
        if set(d['train'].place_id) - set(d['cand']):
            raise ValueError('Training venues missing from frozen catalog')
    d.update(protocol_version=CONFIG['protocol_version'], candidate_mode=mode,
             protocol=f"external_v2/{'external_2023' if external else 'internal_2022'}/{mode}")
    if external:
        d['year'] = '2023'
        d['history'] = pd.read_csv(data_dir/'support.csv', dtype={'user_id':str,'place_id':str})
        d['external_test'] = pd.read_csv(data_dir/'external_targets.csv', dtype={'user_id':str,'place_id':str})
    return d


def provenance(data_dir):
    files = ['interactions.csv','places.csv','users.csv','support.csv','external_targets.csv','support_features.csv']
    return dict(config_sha256=ep.sha256(CONFIG_PATH),
                data_sha256={f:ep.sha256(data_dir/f) for f in files},
                code_sha256={f:ep.sha256(ROOT/f) for f in ['scripts/external_evaluation.py','scripts/cbf.py','scripts/cf.py','scripts/eval_protocol.py','scripts/build_common_dataset.py','configs/experiment.json','requirements.txt']})


def tune(data_dir, out_dir):
    d = load_protocol(data_dir)
    tables = cbf.load_place_tables(data_dir)
    candidates = []
    for variant in ('cbf','cbf_region'):
        grid, best = cbf.tune(d,tables,variant)
        folder = out_dir/'tuning_2022'/variant
        folder.mkdir(parents=True,exist_ok=True)
        grid.to_csv(folder/'validation_grid.csv',index=False)
        score = float(grid.ndcg_at_10_warm.max())
        candidates.append(dict(variant=variant,params=best,validation_ndcg=score))
        print('CBF', variant, best, 'warm validation NDCG',score,flush=True)
    best = max(candidates,key=lambda x:x['validation_ndcg'])
    cf_grid=[]
    for k in CONFIG['cf_k_grid']:
        model=ItemKNN(k).fit(d)
        metrics,_=ep.evaluate(d,model,'validation',top_n=10)
        cf_grid.append(dict(k=k,validation_ndcg=metrics['ndcg_at_10_warm'],metrics=metrics))
        print('CF', k, 'warm validation NDCG',metrics['ndcg_at_10_warm'],flush=True)
    best_cf=max(cf_grid,key=lambda x:x['validation_ndcg'])
    frozen=dict(protocol=CONFIG,provenance=provenance(data_dir),cbf=best,cf_k=best_cf['k'],
                cbf_validation=candidates,cf_validation=cf_grid,
                statement='Selected only on 2022 validation; 2023 targets are not model-selection input.')
    write_json(out_dir/'frozen_settings.json',frozen)
    return frozen


class HistoryCBF:
    """Score new users with frozen 2022 feature fit and their initial history only."""
    def __init__(self,base,history,feature_keys=None):
        self.base,self.history=base,history.copy()
        self.feature_keys=feature_keys or {}
        self.hist={u:g for u,g in self.history.groupby('user_id')}

    def rows(self,g):
        keys=[self.feature_keys.get((u,p),p) for u,p in zip(g.user_id,g.place_id)]
        return self.base.row[keys].to_numpy()

    def score(self,users,candidates):
        h=self.history[self.history.user_id.isin(users)]
        weight=h.rating.copy()
        if self.base.weighting=='centered':
            centered=h.rating-h.groupby('user_id').rating.transform('mean')
            allzero=centered.abs().groupby(h.user_id).transform('max')==0
            weight=centered.where(~allzero,h.rating)
        ui=pd.Series(np.arange(len(users)),index=users)
        W=sp.csr_matrix((weight.to_numpy(float),(ui[h.user_id].to_numpy(),self.rows(h))),
                        shape=(len(users),self.base.X.shape[0]))
        profile=normalize(W@self.base.X)
        return (profile@self.base.X[self.base.row[candidates].to_numpy()].T).toarray()

    def predict(self,users,places):
        out=np.full(len(users),np.nan)
        for i,(user,place) in enumerate(zip(users,places)):
            if user not in self.hist or place not in self.base.row:
                continue
            h=self.hist[user]
            sim=(self.base.X[int(self.base.row[place])]@self.base.X[self.rows(h)].T).toarray().ravel()
            if sim.sum()>0:
                out[i]=np.dot(sim,h.rating.to_numpy())/sim.sum()
        return out


class HistoryCF:
    """Use frozen item similarities; seed venues outside 2022 train cannot contribute."""
    def __init__(self,base,history):
        self.base,self.history=base,history.copy()
        self.hist={u:g[g.place_id.isin(base.item_idx.index)] for u,g in history.groupby('user_id')}

    def matrix(self,users):
        h=self.history[self.history.user_id.isin(users)&self.history.place_id.isin(self.base.item_idx.index)]
        ui=pd.Series(np.arange(len(users)),index=users)
        return sp.csr_matrix((h.rating.to_numpy(float),(ui[h.user_id].to_numpy(),self.base.item_idx[h.place_id].to_numpy())),
                             shape=(len(users),len(self.base.items)))

    def score(self,users,candidates):
        R=self.matrix(users)
        rated=R.copy();rated.data=np.ones_like(rated.data)
        num=(R@self.base.S.T).toarray()
        den=(rated@abs(self.base.S).T).toarray()
        score=np.divide(num,den,out=np.full_like(num,-np.inf),where=den>0)
        result=np.full((len(users),len(candidates)),-np.inf)
        indices=self.base.item_idx.reindex(candidates)
        ok=indices.notna().to_numpy()
        result[:,ok]=score[:,indices.iloc[np.flatnonzero(ok)].to_numpy(int)]
        return result

    def predict(self,users,places):
        result=np.full(len(users),np.nan)
        for i,(user,place) in enumerate(zip(users,places)):
            if user not in self.hist or place not in self.base.item_idx:
                continue
            h=self.hist[user]
            if h.empty:
                continue
            sims=self.base.S[int(self.base.item_idx[place]),self.base.item_idx[h.place_id].to_numpy()].toarray().ravel()
            if sims.sum()>0:
                result[i]=np.dot(sims,h.rating.to_numpy())/sims.sum()
        return result


class AvailabilityFallbackCF:
    """User-level CF routing, not weighted Hybrid: no CF candidates -> popularity."""
    def __init__(self,cf_model,popularity):
        self.cf_model,self.popularity=cf_model,popularity

    def score(self,users,candidates):
        scores=self.cf_model.score(users,candidates)
        # Availability is assessed after excluding input places, as in evaluation.
        history=self.cf_model.history if isinstance(self.cf_model,HistoryCF) else self.cf_model.train
        ui=pd.Series(np.arange(len(users)),index=users)
        ci=pd.Series(np.arange(len(candidates)),index=candidates)
        seen=history[history.user_id.isin(users)&history.place_id.isin(candidates)]
        masked=scores.copy()
        masked[ui[seen.user_id].to_numpy(),ci[seen.place_id].to_numpy()]=-np.inf
        supported=np.isfinite(masked).any(1)
        fallback=self.popularity.score(users,candidates)
        return np.where(supported[:,None],scores,fallback)

    def predict(self,users,places):
        return None  # Routing policy is a ranking method; pure CF reports rating errors.


def support_feature_tables(data_dir,tables):
    rows=pd.read_csv(data_dir/'support_features.csv',dtype=str).fillna('')
    keys={}
    extra={k:{} for k in tables}
    for r in rows.itertuples():
        key=builder.stable_id('initial_feature',r.user_id,r.place_id)
        keys[r.user_id,r.place_id]=key
        address=r.road_address or r.lot_address
        tokens=address.split()
        sido=cbf.sido_key(tokens[0]) if tokens else ''
        values=dict(name=r.place_name,vtype=r.visit_type_code,sido=sido,
                    sgg=f'{sido} {tokens[1]}' if sido and len(tokens)>1 else '')
        for col in tables:
            # For a known place prefer the 2022 snapshot. Unknown initial input
            # features are per-user, never aggregated across 2023 users/targets.
            extra[col][key]=tables[col].get(r.place_id,values[col])
    result={col:pd.concat([series,pd.Series(extra[col],dtype=str)]) for col,series in tables.items()}
    return result,keys


def evaluate_all(data_dir,out_dir):
    frozen=json.loads((out_dir/'frozen_settings.json').read_text())
    if frozen['provenance']!=provenance(data_dir):
        raise ValueError('Data/config/code changed after tuning; freeze settings again before evaluating')
    warm=load_protocol(data_dir)
    tables=cbf.load_place_tables(data_dir)
    base_cbf=cbf.CBF(**frozen['cbf']['params']).fit(warm,tables)
    base_cf=ItemKNN(frozen['cf_k']).fit(warm)
    support=pd.read_csv(data_dir/'support.csv',dtype={'user_id':str,'place_id':str})
    extended,keys=support_feature_tables(data_dir,tables)
    external_base=cbf.CBF(**frozen['cbf']['params']).fit(warm,extended)
    # Fitting the external transform must leave every frozen catalog vector unchanged.
    a=base_cbf.X[base_cbf.row[warm['cand']].to_numpy()]
    b=external_base.X[external_base.row[warm['cand']].to_numpy()]
    delta=a-b
    if delta.nnz and np.abs(delta.data).max()>1e-12:
        raise ValueError('Initial features changed the frozen 2022 feature fit')
    new_cbf=HistoryCBF(external_base,support,keys)
    new_cf=HistoryCF(base_cf,support)
    summary=[]
    for external in (False,True):
        for mode in CONFIG['external_candidate_modes']:
            d=load_protocol(data_dir,mode,external)
            split='external_test' if external else 'test'
            models={name:ep.Baseline(name).fit(warm) for name in ep.CFG['models']['baselines']}
            models['cbf']=new_cbf if external else base_cbf
            models['cf']=new_cf if external else base_cf
            models['cf_or_popularity']=AvailabilityFallbackCF(models['cf'],models['popularity'])
            for name,model in models.items():
                start=time.perf_counter()
                metrics,preds=ep.evaluate(d,model,split,top_n=10)
                if external:
                    targets=d[split]
                    metrics['targets_outside_frozen_catalog']=int((~targets.place_id.isin(set(tables['name'].index))).sum())
                    metrics['support_pairs']=len(support)
                out=ep.save_results(d,name,split,metrics,preds,data_dir,
                    dict(top_n=10,external_config=CONFIG,frozen_settings_sha256=ep.sha256(out_dir/'frozen_settings.json')),
                    time.perf_counter()-start)
                saved=json.loads((out/'metrics.json').read_text())
                summary.append(saved)
                ep.show(f'{name}/{mode}',split,metrics)
            # CF-eligible target-pair supplement uses the SAME subset for every model.
            target=d[split]
            eligible=np.isfinite(models['cf'].predict(target.user_id.to_numpy(),target.place_id.to_numpy()))
            subset=target[eligible].copy()
            if not subset.empty:
                dd=dict(d,eligible_targets=subset,protocol=d['protocol']+'/cf_eligible_pairs')
                for name,model in models.items():
                    metrics,preds=ep.evaluate(dd,model,'eligible_targets',top_n=10)
                    metrics['selection_rule']='target CF prediction supported; same pair set across all models'
                    out=ep.save_results(dd,name,'eligible_targets',metrics,preds,data_dir,
                        dict(top_n=10,external_config=CONFIG,frozen_settings_sha256=ep.sha256(out_dir/'frozen_settings.json')),0.)
                    summary.append(json.loads((out/'metrics.json').read_text()))
    write_json(out_dir/'summary.json',dict(protocol=CONFIG,frozen_settings=frozen,
                quality=json.loads((data_dir/'quality_report.json').read_text()),runs=summary))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','tune','evaluate'],required=True)
    p.add_argument('--raw-dir',type=Path,default=ROOT/'data')
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/proposal_external_v2')
    args=p.parse_args()
    data_dir=args.output_dir/'data'
    if args.stage=='prepare':
        prepare(args.raw_dir,data_dir)
    elif args.stage=='tune':
        tune(data_dir,args.output_dir)
    else:
        evaluate_all(data_dir,args.output_dir)


if __name__=='__main__':
    main()
