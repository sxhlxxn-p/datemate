"""Refit frozen MF and a same-regularization bias-only diagnostic; no retuning."""
import sys,json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_mf
from mf import MatrixFactorization
out={}
for scope in ('internal','external'):
 data_dir=ROOT/('outputs/processed' if scope=='internal' else 'outputs/proposal_external_v2/data')
 s=json.load(open(ROOT/f'outputs/matrix_factorization/{scope}/summary.json'))
 params=s['frozen_settings']['params']
 d=run_mf.load(data_dir,scope)
 mf=MatrixFactorization(**params).fit(d)
 bias=MatrixFactorization(**dict(params,factors=0)).fit(d)
 t=d['train']; users=t.user_id.to_numpy();items=t.place_id.to_numpy()
 ui=mf.user_idx.loc[users].to_numpy(int);ii=mf.item_idx.loc[items].to_numpy(int)
 latent=np.einsum('ij,ij->i',mf.P[ui],mf.Q[ii])
 delta=mf.predict(users,items)-bias.predict(users,items)
 out[scope]={'latent_absolute_max_on_train_pairs':float(np.abs(latent).max()),'latent_absolute_mean_on_train_pairs':float(np.abs(latent).mean()),'user_factor_frobenius_norm':float(np.linalg.norm(mf.P)),'item_factor_frobenius_norm':float(np.linalg.norm(mf.Q)),'same_regularization_bias_prediction_max_difference':float(np.abs(delta).max()),'diagnostic_params':params}
print(json.dumps(out,indent=2))
(ROOT/'outputs/matrix_factorization/latent_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
