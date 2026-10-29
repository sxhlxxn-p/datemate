"""Validate generated keys, referential integrity and the preserved evaluation splits."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def validate(root):
    rows = read(root / 'interactions.csv')
    places = read(root / 'places.csv')
    users = read(root / 'users.csv')
    assert rows and places and users, 'Empty dataset'
    for table, key in ((rows, 'interaction_id'), (places, 'place_id'), (users, 'user_id')):
        assert len({r[key] for r in table}) == len(table), f'Duplicate {key}'
    pids = {r['place_id'] for r in places}
    uids = {r['user_id'] for r in users}
    pairs = defaultdict(set)
    by_user = defaultdict(dict)
    train_items = defaultdict(set)
    for r in rows:
        assert r['place_id'] in pids and r['user_id'] in uids, 'Broken reference'
        assert r['rating'] in {'1', '2', '3', '4', '5'}, 'Invalid rating'
        assert r['user_id'].startswith(r['year'] + ':'), 'User year namespace'
        assert r['temporal_split'] == ('train' if r['year'] == '2022' else 'test'), 'Temporal split'
        split = r['known_user_split']
        assert split in {'train', 'train_only', 'validation', 'test'}, 'Unknown split'
        pairs[(r['user_id'], r['place_id'])].add(split)
        by_user[r['user_id']][r['place_id']] = split
        if split in {'train', 'train_only'}:
            train_items[r['year']].add(r['place_id'])
    assert all(len(s) == 1 for s in pairs.values()), 'User-place leakage'
    for assignments in by_user.values():
        values = list(assignments.values())
        if len(values) < 5:
            assert set(values) == {'train_only'}, 'Short-history holdout'
        else:
            assert values.count('test') == values.count('validation') == 1, 'Holdout count'
            assert values.count('train') == len(values) - 2, 'Train count'
    for r in rows:
        expected = str(int(r['place_id'] in train_items[r['year']]))
        assert r['is_warm_item_in_known_user_train'] == expected, 'Warm flag mismatch'
    report = {'status': 'PASS', 'interactions': len(rows), 'places': len(places),
              'users': len(users), 'pairs': len(pairs)}
    print(json.dumps(report))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('outputs/processed'))
    validate(parser.parse_args().input_dir)
