"""Fuse A/B prediction CSVs; no model training or evaluation is performed."""
import argparse
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datemate.hybrid import combine_scores, couple_recommendations


def read(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows):
    if not rows:
        raise ValueError('No common eligible candidates to write')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cbf', required=True)
    p.add_argument('--cf', required=True)
    p.add_argument('--alpha', type=float, required=True)
    p.add_argument('--cbf-bounds', type=float, nargs=2, default=(-1, 1))
    p.add_argument('--cf-bounds', type=float, nargs=2, default=(1, 5))
    p.add_argument('--output', required=True)
    p.add_argument('--couple', nargs=2)
    p.add_argument('--interactions', help='Required for real couple demo: exclude both train histories')
    p.add_argument('--synthetic-demo', action='store_true')
    p.add_argument('--k', type=int, default=10)
    args = p.parse_args()
    rows = combine_scores(read(args.cbf), read(args.cf), args.alpha,
                          args.cbf_bounds, args.cf_bounds)
    if args.couple and not args.interactions and not args.synthetic_demo:
        p.error('--couple requires --interactions (or --synthetic-demo for fictitious data)')
    write(args.output, rows)
    if args.couple:
        excluded = set()
        if args.interactions:
            scopes = {(r['year'], r['split']) for r in rows if r['user_id'] in args.couple}
            if len(scopes) != 1:
                raise ValueError('Select a single known-user year/split for demo')
            year, split = next(iter(scopes))
            if split not in ('validation', 'test'):
                raise ValueError('Automatic history exclusion supports known-user validation/test only')
            excluded = {r['place_id'] for r in read(args.interactions)
                        if r['user_id'] in args.couple and r['year'] == year
                        and r['known_user_split'] in ('train', 'train_only')}
        for method in ('mean', 'min'):
            ranked = couple_recommendations(rows, *args.couple, method, args.k, excluded)
            out = str(Path(args.output).with_suffix('')) + f'_couple_{method}.csv'
            write(out, ranked)
            print(f'{method}: {out}')
            for r in ranked:
                print(f"  {r['place_id']}: {r['score']:.3f} (A={r['user_a_score']:.3f}, B={r['user_b_score']:.3f})")
    print(f'Hybrid scores: {args.output}; rows={len(rows)}')


if __name__ == '__main__':
    main()
