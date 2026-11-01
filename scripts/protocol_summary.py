"""Print model-independent evaluation cohort counts; never exports user records."""
import argparse
import json
from pathlib import Path
import eval_protocol as ep


def summarize(d, split):
    ev = d[split]
    relevant = ev[ev.rating >= ep.REL]
    warm = relevant[relevant.place_id.isin(set(d["train"].place_id))]
    return dict(year=d["year"], split=split, candidate_mode=d["candidate_mode"],
                train_pairs=len(d["train"]), candidates=len(d["cand"]),
                eval_pairs=len(ev), eval_users=ev.user_id.nunique(),
                positive_users=relevant.user_id.nunique(), warm_positive_users=warm.user_id.nunique(),
                relevant_pairs=len(relevant), relevant_warm=len(warm),
                relevant_cold=len(relevant)-len(warm),
                relevant_in_candidates=int(relevant.place_id.isin(set(d["cand"])).sum()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=ep.ROOT / "outputs/processed")
    p.add_argument("--year", default="2022")
    args = p.parse_args()
    result = [summarize(ep.load_data(args.year, args.input_dir, mode), split)
              for mode in ("warm", "all") for split in ("validation", "test")]
    print(json.dumps(result, ensure_ascii=False, indent=2))
