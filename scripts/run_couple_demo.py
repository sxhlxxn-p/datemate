"""DateMate: real two-user couple recommendation demo (no retuning/test evaluation).

Place this file in scripts/run_couple_demo.py and run from repository root.
Uses the alpha selected on warm validation and the existing Role C fusion logic.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import eval_protocol as ep
from cbf import CBF, load_place_tables
from cf import ItemKNN
from run_hybrid_real import Hybrid
from datemate.hybrid import couple_recommendations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', default='2022')
    parser.add_argument('--input-dir', type=Path, default=ep.ROOT / 'outputs' / 'processed')
    parser.add_argument('--output-dir', type=Path, default=ep.ROOT / 'outputs' / 'couple_demo')
    parser.add_argument('--tuning-file', type=Path, default=None)
    parser.add_argument('--users', nargs=2, metavar=('USER_A', 'USER_B'), help='Two user IDs present in the training data')
    parser.add_argument('--candidate-mode', choices=('warm', 'all'), default='all')
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    if args.k < 1:
        parser.error('--k must be positive')

    tuning_file = args.tuning_file or (ep.ROOT / 'outputs' / 'hybrid_real' / f'{args.year}_alpha_tuning.json')
    tuning = json.loads(tuning_file.read_text(encoding='utf-8'))
    if str(tuning.get('year')) != str(args.year):
        parser.error('Tuning year differs from requested year')
    if tuning.get('selection') != 'warm validation ndcg_at_10_warm':
        parser.error('Expected alpha chosen on warm validation')
    alpha = float(tuning['best_alpha'])
    cf_k = int(tuning['cf_k'])
    cbf_params = tuning['cbf']

    warm = ep.load_data(args.year, args.input_dir, 'warm')
    catalog = ep.load_data(args.year, args.input_dir, args.candidate_mode)
    train = warm['train']
    known = sorted(train.user_id.unique())
    if len(known) < 2:
        raise ValueError('Need at least two training users')
    users = list(args.users) if args.users else known[:2]
    if users[0] == users[1] or any(u not in set(known) for u in users):
        parser.error('Select two distinct user IDs with training histories')

    print(f'Users: {users[0]}, {users[1]} | alpha={alpha:.2f} | candidates={args.candidate_mode}', flush=True)
    tables = load_place_tables(args.input_dir)
    cbf = CBF(**cbf_params).fit(warm, tables)
    cf = ItemKNN(k=cf_k).fit(warm)
    cand = np.asarray(catalog['cand'])
    score_matrix = Hybrid(cbf, cf, alpha).score(np.asarray(users), cand)

    # Both partners' training visits are excluded for both ranking methods.
    seen = set(train.loc[train.user_id.isin(users), 'place_id'])
    rows = [dict(user_id=u, place_id=str(place), year=str(args.year), split='demo', score=float(score_matrix[i, j]))
            for i, u in enumerate(users) for j, place in enumerate(cand)]

    places = pd.read_csv(args.input_dir / 'places.csv', dtype=str)
    if 'place_name' in places.columns:
        names = dict(zip(places.place_id, places.place_name.fillna('')))
    else:
        names = {}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for method in ('mean', 'min'):
        result = couple_recommendations(rows, users[0], users[1], method=method, k=args.k, excluded=seen)
        for rank, row in enumerate(result, 1):
            row['rank'] = rank
            row['place_name'] = names.get(row['place_id'], '')
        cols = ['rank', 'place_id', 'place_name', 'user_a_score', 'user_b_score', 'score', 'method']
        path = args.output_dir / f'{args.year}_{args.candidate_mode}_couple_{method}.csv'
        pd.DataFrame(result, columns=cols).to_csv(path, index=False, encoding='utf-8-sig')
        print(f'\n{method.upper()} Top {len(result)} -> {path}', flush=True)
        for r in result:
            print(f" {r['rank']:>2}. {r['place_name'] or r['place_id']} | couple={r['score']:.4f} "
                  f"A={r['user_a_score']:.4f} B={r['user_b_score']:.4f}", flush=True)

    metadata = {'year': str(args.year), 'user_a': users[0], 'user_b': users[1],
                'alpha': alpha, 'cf_k': cf_k, 'candidate_mode': args.candidate_mode,
                'candidates': len(cand), 'excluded_training_places': len(seen),
                'note': 'Demonstration only; no couple satisfaction ground truth.'}
    (args.output_dir / 'demo_info.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
