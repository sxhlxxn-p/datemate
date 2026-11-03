"""Tune on 2022 validation, freeze manifests, evaluate internal/external separately."""
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import eval_protocol as ep
import external_evaluation as external
from mf import MatrixFactorization

ROOT = ep.ROOT
CONFIG_PATH = ROOT/'configs/matrix_factorization.json'
CONFIG = json.loads(CONFIG_PATH.read_text())


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def provenance(data_dir, protocol):
    names = ['interactions.csv','places.csv','users.csv']
    if protocol=='external': names += ['support.csv','external_targets.csv']
    files = ['scripts/mf.py','scripts/run_mf.py','scripts/eval_protocol.py','scripts/external_evaluation.py',
             'configs/experiment.json','configs/external_evaluation_v2.json','configs/matrix_factorization.json','requirements.txt']
    return {'data_sha256':{p:ep.sha256(data_dir/p) for p in names},
            'code_sha256':{p:ep.sha256(ROOT/p) for p in files}}


def load(data_dir, protocol, mode='warm', new_users=False):
    if protocol=='internal': return ep.load_data('2022',data_dir,mode)
    return external.load_protocol(data_dir, 'warm_2022' if mode=='warm' else 'catalog_2022',new_users)


def tune(data_dir, protocol, output):
    d=load(data_dir,protocol)
    rows=[]
    for factors,reg in itertools.product(CONFIG['factors_grid'],CONFIG['regularization_grid']):
        start=time.perf_counter()
        params=dict(factors=factors,reg=reg,epochs=CONFIG['epochs'],seed=CONFIG['seed'])
        model=MatrixFactorization(**params).fit(d)
        metrics,_=ep.evaluate(d,model,'validation',top_n=10)
        value=metrics['ndcg_at_10_warm']
        if value is None: raise ValueError('No warm validation positives')
        rows.append(dict(params=params,metrics=metrics,loss_history=model.loss_history,
                         seconds=round(time.perf_counter()-start,2)))
        print(protocol,params,'validation warm NDCG',value,flush=True)
    best=sorted(rows,key=lambda r:(-r['metrics']['ndcg_at_10_warm'],r['params']['factors'],r['params']['reg']))[0]
    frozen=dict(protocol=protocol,selection=CONFIG['selection'],params=best['params'],validation_grid=rows,
                provenance=provenance(data_dir,protocol))
    write(output/'frozen_settings.json',frozen)
    return frozen


def evaluate(data_dir,protocol,output):
    frozen=json.loads((output/'frozen_settings.json').read_text())
    if frozen['protocol']!=protocol or frozen['provenance']!=provenance(data_dir,protocol):
        raise ValueError('Data or source changed since validation tuning')
    warm=load(data_dir,protocol)
    base=MatrixFactorization(**frozen['params']).fit(warm)
    bias=MatrixFactorization(**CONFIG['bias_only_ablation'],epochs=CONFIG['epochs'],seed=CONFIG['seed']).fit(warm)
    runs=[]
    contexts=(False,True) if protocol=='external' else (False,)
    for new_users in contexts:
        for mode in ('warm','all'):
            d=load(data_dir,protocol,mode,new_users)
            # Separate output scope preserves every previously generated experiment.
            d['protocol']='mf_extension/'+d['protocol']
            split='external_test' if new_users else 'test'
            mf=base.with_history(d['history']) if new_users else base
            b=bias.with_history(d['history']) if new_users else bias
            popular=ep.Baseline('popularity').fit(warm)
            models={name:ep.Baseline(name).fit(warm) for name in ('global_mean','item_mean')}
            models.update(popularity=popular,mf=mf,bias_only=b,mf_or_popularity=external.AvailabilityFallbackCF(mf,popular))
            for name,model in models.items():
                start=time.perf_counter()
                metrics,predictions=ep.evaluate(d,model,split,top_n=10)
                path=ep.save_results(d,name,split,metrics,predictions,data_dir,
                     {'top_n':10,'mf_config':CONFIG,'selected_params':frozen['params'],
                      'frozen_settings_sha256':ep.sha256(output/'frozen_settings.json')},time.perf_counter()-start)
                runs.append(json.loads((path/'metrics.json').read_text()))
                ep.show(name+'/'+protocol+'/'+mode,split,metrics)
            # Compare supported target pairs with identical baselines, not different denominators.
            targets=d[split]
            mask=np.isfinite(mf.predict(targets.user_id.to_numpy(),targets.place_id.to_numpy()))
            if mask.any():
                dd=dict(d,eligible_targets=targets[mask].copy(),protocol=d['protocol']+'/mf_eligible_pairs')
                for name in ('global_mean','item_mean','popularity','mf','bias_only'):
                    metrics,predictions=ep.evaluate(dd,models[name],'eligible_targets',top_n=10)
                    path=ep.save_results(dd,name,'eligible_targets',metrics,predictions,data_dir,
                        {'top_n':10,'selection_rule':'MF-supported target pairs; identical subset for all models'},0.)
                    runs.append(json.loads((path/'metrics.json').read_text()))
    summary={'scope':protocol,'frozen_settings':frozen,'runs':runs}
    # Reject comparisons that accidentally use different candidate/target/user manifests.
    grouped={}
    for r in runs: grouped.setdefault(r['meta']['protocol'],[]).append(r)
    for group in grouped.values():
        for key in ('candidate_sha256','evaluation_pairs_sha256','positive_users_sha256','warm_positive_users_sha256','data_sha256'):
            if len({json.dumps(r['meta'][key],sort_keys=True) for r in group})!=1:
                raise ValueError('Model comparison input mismatch: '+key)
    summary['matching_input_manifests']=True
    write(output/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['tune','evaluate'],required=True)
    p.add_argument('--protocol',choices=['internal','external'],default='internal')
    p.add_argument('--input-dir',type=Path)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/matrix_factorization')
    args=p.parse_args()
    data_dir=args.input_dir or ROOT/('outputs/processed' if args.protocol=='internal' else 'outputs/proposal_external_v2/data')
    output=args.output_dir/args.protocol
    if args.stage=='tune': tune(data_dir,args.protocol,output)
    else: evaluate(data_dir,args.protocol,output)


if __name__=='__main__': main()
