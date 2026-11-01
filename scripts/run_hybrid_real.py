"""DateMate C: actual CBF+CF hybrid evaluation (2022 known-user protocol v1.1).

Place in scripts/run_hybrid_real.py and run from repository root.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_protocol as ep
from cbf import CBF, load_place_tables
from cf import ItemKNN


class Hybrid:
    """Fixed-scale fusion; unavailable CF scores fall back to CBF."""

    def __init__(self, cbf, cf, alpha):
        self.cbf, self.cf, self.alpha = cbf, cf, float(alpha)
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")

    def score(self, users, cand):
        c = ep.checked_scores(self.cbf, users, cand)
        f = ep.checked_scores(self.cf, users, cand)
        if not np.isfinite(c).all() or (c < -1-1e-8).any() or (c > 1+1e-8).any():
            raise ValueError("CBF cosine score outside [-1, 1] or non-finite")
        valid = np.isfinite(f)
        if ((f[valid] < 1-1e-8) | (f[valid] > 5+1e-8)).any():
            raise ValueError("CF score outside [1, 5]")
        cn = np.clip((c + 1) / 2, 0, 1)
        # Compute only supported cells; -inf is never normalized.
        cn[valid] = ((1 - self.alpha) * cn[valid]
                     + self.alpha * np.clip((f[valid] - 1) / 4, 0, 1))
        return cn

    def predict(self, users, places):
        # Ranking fusion is not calibrated as a 1-5 rating predictor.
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--year', default='2022')
    p.add_argument('--input-dir', type=Path, default=ep.ROOT / 'outputs' / 'processed')
    p.add_argument('--output-dir', type=Path, default=ep.ROOT / 'outputs' / 'hybrid_real')
    p.add_argument('--final-test', action='store_true', help='Evaluate held-out test only after alpha is fixed')
    p.add_argument('--cf-k', type=int, default=40)
    args = p.parse_args()

    warm = ep.load_data(args.year, args.input_dir, 'warm')
    tables = load_place_tables(args.input_dir)
    cbf = CBF(name_mode='char', w_type=0.5, w_region=0.5, weighting='raw').fit(warm, tables)
    cf = ItemKNN(k=args.cf_k).fit(warm)
    alphas = [round(float(x), 2) for x in np.linspace(0, 1, 11)]
    scores = []
    print('Tuning alpha on warm validation (NDCG@10 warm)...', flush=True)
    for alpha in alphas:
        metrics, _ = ep.evaluate(warm, Hybrid(cbf, cf, alpha), 'validation', top_n=10)
        scores.append({'alpha': alpha,
                       'ndcg_at_10_warm': metrics['ndcg_at_10_warm'],
                       'recall_at_10_warm': metrics['recall_at_10_warm']})
        print(f"alpha={alpha:.1f} warm_NDCG@10={metrics['ndcg_at_10_warm']:.6f}", flush=True)
    # Deterministic tie break: smaller alpha wins.
    best = max(scores, key=lambda r: (r['ndcg_at_10_warm'], -r['alpha']))['alpha']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f'{args.year}_alpha_tuning.json').write_text(
        json.dumps({'year': args.year, 'cf_k': args.cf_k,
                    'cbf': {'name_mode': 'char', 'w_type': 0.5, 'w_region': 0.5, 'weighting': 'raw'},
                    'selection': 'warm validation ndcg_at_10_warm',
                    'best_alpha': best, 'results': scores}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Chosen alpha={best:.2f}', flush=True)
    for mode in ('warm', 'all'):
        d = ep.load_data(args.year, args.input_dir, mode)
        for split in (('validation', 'test') if args.final_test else ('validation',)):
            metrics, _ = ep.evaluate(d, Hybrid(cbf, cf, best), split, top_n=10)
            out = args.output_dir / f'{args.year}_{mode}_{split}_metrics.json'
            out.write_text(json.dumps({'year': args.year, 'candidate_mode': mode,
                                       'split': split, 'alpha': best,
                                       'cf_k': args.cf_k, 'metrics': metrics},
                                      ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"{mode}/{split}: recall@10={metrics['recall_at_10']:.6f} "
                  f"ndcg@10={metrics['ndcg_at_10']:.6f} "
                  f"warm_ndcg@10={metrics['ndcg_at_10_warm']:.6f}", flush=True)
    print(f'Results saved: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
