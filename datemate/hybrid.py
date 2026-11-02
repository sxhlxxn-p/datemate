"""Role C: score fusion. Inputs must use a shared candidate catalog.

CBF cosine scores are mapped from [-1, 1] to [0, 1]; CF rating
predictions from [1, 5] to [0, 1]. No statistics are fitted on holdouts.
These fixed mappings are a starting contract, not learned calibration.
"""
import math


def normalize(value, bounds):
    """Map a finite score in explicit bounds to [0, 1]; reject bad data."""
    value = float(value)
    low, high = bounds
    if not math.isfinite(value) or high <= low or not low <= value <= high:
        raise ValueError(f"Score {value} is outside finite bounds {bounds}")
    return (value - low) / (high - low)


def combine_scores(cbf_rows, cf_rows, alpha=0.5, cbf_bounds=(-1, 1), cf_bounds=(1, 5)):
    """Fuse rows keyed by user_id/place_id/year/split, preserving CBF coverage.

    Each row has score, user_id, place_id, year, split. An absent/blank CF
    score means unsupported. B must omit unsupported predictions rather
    than supply its baseline fallback. CBF must cover all shared candidates.
    """
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    def indexed(rows):
        result = {}
        for row in rows:
            key = tuple(str(row[k]) for k in ('user_id', 'place_id', 'year', 'split'))
            if key in result:
                raise ValueError(f"Duplicate prediction key: {key}")
            result[key] = row
        return result
    cbf, cf = indexed(cbf_rows), indexed(cf_rows)
    if cf.keys() - cbf.keys():
        raise ValueError("CF includes candidates missing from CBF")
    output = []
    for key in sorted(cbf):
        c = normalize(cbf[key]['score'], cbf_bounds)
        raw = cf.get(key, {}).get('score')
        supported = raw is not None and str(raw).strip() != ''
        f = normalize(raw, cf_bounds) if supported else None
        score = alpha * f + (1 - alpha) * c if supported else c
        output.append(dict(zip(('user_id', 'place_id', 'year', 'split'), key),
                           score=score, cbf_score=c, cf_score=f,
                           cf_available=int(supported), alpha=alpha,
                           score_source='hybrid' if supported else 'cbf_fallback'))
    return output


def couple_recommendations(rows, user_a, user_b, method='mean', k=10, excluded=()):
    """Rank shared candidates for one year/split; exclude either user's seen items.

    Caller supplies the union of both train histories in excluded. Same-scope
    rows only: merging protocols or holdouts is an error. Pair scores are a
    demonstration, not evidence of actual joint satisfaction.
    """
    if user_a == user_b or method not in ('mean', 'min') or k < 1:
        raise ValueError('Use two distinct users, mean/min, and positive k')
    selected = [r for r in rows if r['user_id'] in (user_a, user_b)]
    if len({(str(r['year']), r['split']) for r in selected}) > 1:
        raise ValueError('Couple demo requires exactly one year/split')
    by_user = {user_a: {}, user_b: {}}
    for row in selected:
        target = by_user[row['user_id']]
        if row['place_id'] in target:
            raise ValueError('Duplicate user/place in couple scores')
        target[row['place_id']] = normalize(row['score'], (0, 1))
    if not all(by_user.values()):
        raise ValueError('Both users need candidate scores')
    blocked = set(excluded)
    result = []
    for place in by_user[user_a].keys() & by_user[user_b].keys() - blocked:
        a, b = by_user[user_a][place], by_user[user_b][place]
        result.append(dict(place_id=place, user_a_score=a, user_b_score=b,
                           score=(a+b)/2 if method == 'mean' else min(a,b), method=method))
    return sorted(result, key=lambda r: (-r['score'], r['place_id']))[:k]
