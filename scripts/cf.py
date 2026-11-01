from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

import eval_protocol as ep

ROOT = ep.ROOT


class ItemKNN:
    """Item-based KNN Collaborative Filtering."""

    def __init__(self, k=20):
        self.k = int(k)

    def fit(self, d):
        train = d["train"].copy()
        self.train = train
        self.cand = d["cand"]

        self.users = np.array(sorted(train.user_id.unique()))
        self.items = np.array(sorted(train.place_id.unique()))
        self.user_idx = pd.Series(np.arange(len(self.users)), index=self.users)
        self.item_idx = pd.Series(np.arange(len(self.items)), index=self.items)

        rows = self.user_idx[train.user_id].to_numpy()
        cols = self.item_idx[train.place_id].to_numpy()
        vals = train.rating.to_numpy(float)

        self.R = sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(len(self.users), len(self.items)),
        )

        item_user = self.R.T.tocsr()
        n_neighbors = min(self.k + 1, len(self.items))

        knn = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=n_neighbors,
        )
        knn.fit(item_user)
        distances, neighbors = knn.kneighbors(item_user)

        sim_rows, sim_cols, sim_vals = [], [], []
        for item_i in range(len(self.items)):
            for dist, item_j in zip(distances[item_i], neighbors[item_i]):
                if item_i == item_j:
                    continue
                similarity = 1.0 - float(dist)
                if similarity <= 0:
                    continue
                sim_rows.append(item_i)
                sim_cols.append(item_j)
                sim_vals.append(similarity)

        self.S = sp.csr_matrix(
            (sim_vals, (sim_rows, sim_cols)),
            shape=(len(self.items), len(self.items)),
        )
        return self

    def score(self, users, cand):
        users = np.asarray(users)
        cand = np.asarray(cand)

        user_rows = self.user_idx.reindex(users)
        result = np.full((len(users), len(cand)), -np.inf, dtype=float)

        valid_users = ~user_rows.isna()
        if not valid_users.any():
            return result

        valid_user_pos = np.where(valid_users.to_numpy())[0]
        train_user_idx = user_rows[valid_users].astype(int).to_numpy()
        Rsub = self.R[train_user_idx]

        numerator = Rsub @ self.S.T
        rated = Rsub.copy()
        rated.data = np.ones_like(rated.data)
        denominator = rated @ abs(self.S).T

        numerator = numerator.toarray()
        denominator = denominator.toarray()

        scores = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, -np.inf, dtype=float),
            where=denominator > 0,
        )

        candidate_idx = self.item_idx.reindex(cand)
        valid_items = ~candidate_idx.isna()

        if valid_items.any():
            valid_item_pos = np.where(valid_items.to_numpy())[0]
            train_item_idx = candidate_idx[valid_items].astype(int).to_numpy()
            result[np.ix_(valid_user_pos, valid_item_pos)] = scores[:, train_item_idx]

        return result

    def predict(self, users, places):
        out = np.full(len(users), np.nan, dtype=float)

        for i, (user, place) in enumerate(zip(users, places)):
            if user not in self.user_idx.index or place not in self.item_idx.index:
                continue

            u = int(self.user_idx[user])
            j = int(self.item_idx[place])

            rated_items = self.R[u].indices
            ratings = self.R[u].data
            if len(rated_items) == 0:
                continue

            sims = self.S[j, rated_items].toarray().ravel()
            mask = sims > 0
            if not mask.any():
                continue

            out[i] = np.sum(sims[mask] * ratings[mask]) / np.sum(sims[mask])

        return out


def evaluate_one(d, input_dir, k, split, predictions_top=100, save=True):
    start = time.perf_counter()
    model = ItemKNN(k=k).fit(d)
    metrics, predictions = ep.evaluate(d, model, split, top_n=predictions_top)
    seconds = time.perf_counter() - start

    name = f"item_knn_k{k}"
    ep.show(name, split, metrics)

    if save:
        out = ep.save_results(
            d,
            name,
            split,
            metrics,
            predictions,
            input_dir,
            {
                "kind": "item_knn_cf",
                "k": k,
                "metric": "cosine",
                "algorithm": "brute",
                "predictions_top": predictions_top,
            },
            seconds,
        )
        print(f"  saved: {out.relative_to(ROOT).as_posix()}")

    return metrics, seconds


def tune_k(d, input_dir, year, ks, predictions_top=100):
    rows = []
    print("\n=== Item-KNN validation tuning ===")

    for k in ks:
        print(f"\n--- k={k} ---")
        metrics, seconds = evaluate_one(
            d, input_dir, k, "validation", predictions_top, save=True
        )
        rows.append({
            "k": k,
            "recall_at_10": metrics["recall_at_10"],
            "ndcg_at_10": metrics["ndcg_at_10"],
            "recall_at_10_warm": metrics["recall_at_10_warm"],
            "ndcg_at_10_warm": metrics["ndcg_at_10_warm"],
            "recommendable_user_ratio": metrics["recommendable_user_ratio"],
            "item_coverage": metrics["item_coverage"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mae_supported": metrics["mae_supported"],
            "rmse_supported": metrics["rmse_supported"],
            "rating_supported_pairs": metrics["rating_supported_pairs"],
            "seconds": round(seconds, 3),
        })

    result = pd.DataFrame(rows).sort_values(
        ["ndcg_at_10", "recall_at_10", "k"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    best_k = int(result.iloc[0]["k"])

    out_dir = ROOT / "outputs" / "experiments" / f"known_user_{year}" / "item_knn"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tuning_validation.csv"
    result.to_csv(path, index=False, encoding="utf-8")

    print("\n=== validation summary ===")
    print(result.round(4).to_string(index=False))
    print(f"\nchosen k={best_k} by validation NDCG@10")
    print(f"saved: {path.relative_to(ROOT).as_posix()}")

    return best_k


def main():
    ap = argparse.ArgumentParser(description="DateMate B-part Item-based KNN CF")
    ap.add_argument("--year", default="2022")
    ap.add_argument("--input-dir", type=Path, default=ROOT / "outputs" / "processed")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--ks", nargs="+", type=int, default=[10, 20, 40])
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--predictions-top", type=int, default=100)
    args = ap.parse_args()

    print("Loading data...")
    d = ep.load_data(args.year, args.input_dir)
    print(
        f"year={args.year}, train={len(d['train']):,}, "
        f"validation={len(d['validation']):,}, test={len(d['test']):,}, "
        f"candidates={len(d['cand']):,}"
    )

    if args.tune:
        best_k = tune_k(d, args.input_dir, args.year, args.ks, args.predictions_top)
        if args.final_test:
            print(f"\n=== final test with chosen k={best_k} ===")
            evaluate_one(
                d, args.input_dir, best_k, "test", args.predictions_top, save=True
            )
        return

    print(f"\nTraining Item-KNN (k={args.k})...")
    evaluate_one(
        d, args.input_dir, args.k, args.split, args.predictions_top, save=True
    )


if __name__ == "__main__":
    main()
