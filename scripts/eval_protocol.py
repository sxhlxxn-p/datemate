"""
임시 평가 + 기준선 3종 (docs/evaluation.md 규칙 반영).
공식 모델 평가 코드 올라오면 교체 예정

하는 일
  1. outputs/processed/interactions.csv 에서 known_user_split 으로 train / validation / test 생성.
  2. 모델이 (사용자 x 후보 장소) 점수표를 주면 NDCG@10, Recall@10 (전체/warm-only), 추천 가능 비율, coverage 계산.
  3. 평점 예측 MAE/RMSE (예측값 [1,5] 고정, fallback 포함 전체 / 지원 가능한 쌍만) 를 계산.
  4. outputs/experiments/known_user_<연도>/<모델>/<분할>/ 에 config.json, metrics.json, predictions.csv, environment.txt 저장.

다른 파일(scripts/cbf.py 등)에서 가져다 쓰는 법:
  from eval_protocol import load_data, evaluate, save_results
  모델은 score(users, cand) -> (사용자 수, 후보 수) 배열 과 predict(users, places) -> 배열 또는 None 만 있으면 됨.

사용: python -X utf8 scripts/eval_protocol.py --split validation
      python -X utf8 scripts/eval_protocol.py --split test          (test 는 최종 비교용, 모델당 한 번만)
      python -X utf8 scripts/eval_protocol.py --self-check          (평가 자체 점검)
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
K = CFG["k"]                      # 10
REL = CFG["relevance_rating_gte"]  # 4 : 평점 4 이상이면 정답
# 위치 i(0부터)에 있는 항목의 할인 계수 1/log2(i+2) : NDCG 에 사용
DISC = 1.0 / np.log2(np.arange(2, K + 2))


# ------------------------------------------------------------------ 데이터 준비
def load_data(year, input_dir):
    """한 해의 train / validation / test 사용자-장소 쌍(평균 평점)과 후보 장소 목록을 만듦."""
    cols = {"year": str, "user_id": str, "place_id": str}
    g = pd.read_csv(input_dir / "interactions.csv", dtype=cols)
    g = g[g.year == year]

    def pairs(labels):
        # 분할 안에서만 같은 사용자-장소의 평점을 평균 냄 (pair_aggregation = mean_within_split)
        p = g[g.known_user_split.isin(labels)]
        return p.groupby(["user_id", "place_id"], as_index=False)["rating"].mean()

    train = pairs(CFG["train_labels"])
    val = pairs([CFG["tune_label"]])
    test = pairs([CFG["final_label"]])
    # 같은 사용자-장소는 한 분할에만 있어야 함 (누수 점검)
    key = lambda p: set(zip(p.user_id, p.place_id))
    assert not (key(train) & key(val) or key(train) & key(test) or key(val) & key(test)), "pair in two splits"
    # 후보 = 같은 해 train/train_only 에 있는 장소 전체. 사전순으로 정렬해 두면
    # '열 순서 = place_id 사전순' 이 되어 안정 정렬만으로 동점 규칙이 지켜짐.
    cand = np.array(sorted(train.place_id.unique()))
    return dict(year=year, train=train, validation=val, test=test, cand=cand,
                global_mean=float(train.rating.mean()))


# ------------------------------------------------------------------ 기준선
class Baseline:
    """global_mean / item_mean / popularity (평가 문서의 기준선 3종)."""

    def __init__(self, kind):
        self.kind = kind

    def fit(self, d):
        t = d["train"]
        self.gm = d["global_mean"]
        self.item_mean = t.groupby("place_id").rating.mean()   # train 쌍 평균 평점 (장소별)
        self.pop = t.groupby("place_id").size().astype(float)  # train 사용자-장소 쌍 수 (인기)
        return self

    def score(self, users, cand):
        if self.kind == "global_mean":      # 전부 같은 점수 -> 순위는 place_id 사전순
            v = np.full(len(cand), self.gm)
        elif self.kind == "item_mean":      # 장소 평균 평점이 높은 순
            v = self.item_mean.reindex(cand).fillna(self.gm).to_numpy()
        else:                               # popularity: 방문 쌍이 많은 순
            v = self.pop.reindex(cand).to_numpy()
        return np.tile(v, (len(users), 1))

    def predict(self, users, places):
        """평점 예측. 못 맞히는 쌍은 NaN (평가기가 global mean 으로 대체). popularity 는 평점 예측이 없음."""
        if self.kind == "popularity":
            return None
        if self.kind == "global_mean":
            return np.full(len(places), self.gm)
        return self.item_mean.reindex(places).to_numpy()


# ------------------------------------------------------------------ 평가
def evaluate(d, model, split, top_n=100):
    """모델 하나를 한 분할(validation 또는 test)에서 평가함. (metrics dict, predictions DataFrame) 반환."""
    ev = d[split]
    users = np.array(sorted(ev.user_id.unique()))        # 평가 사용자 = 이 분할에 쌍이 있는 사용자
    cand = d["cand"]
    S = np.asarray(model.score(users, cand), dtype=float)
    assert S.shape == (len(users), len(cand)) and not np.isnan(S).any(), "bad score matrix"
    u_idx = pd.Series(np.arange(len(users)), index=users)
    c_idx = pd.Series(np.arange(len(cand)), index=cand)

    # 후보에서 그 사용자의 train 장소 제외 (-inf = 추천 불가)
    t = d["train"][d["train"].user_id.isin(users)]
    S[u_idx[t.user_id].to_numpy(), c_idx[t.place_id].to_numpy()] = -np.inf

    # 정답: 평점 >= 4. 후보 안(warm) 정답은 R 표에 표시, 후보 밖(cold) 정답은 개수만 셈.
    rel = ev[ev.rating >= REL]
    rel_total = rel.groupby("user_id").size().reindex(users, fill_value=0).to_numpy()
    warm = rel[rel.place_id.isin(set(cand))]
    R = np.zeros(S.shape, dtype=bool)
    R[u_idx[warm.user_id].to_numpy(), c_idx[warm.place_id].to_numpy()] = True
    rel_warm = R.sum(1)

    # 점수 높은 순. 안정 정렬이라 같은 점수는 열 순서(= place_id 사전순)가 유지됨.
    order = np.argsort(-S, axis=1, kind="stable")[:, :max(K, top_n)]
    top = order[:, :K]
    hit = np.take_along_axis(R, top, 1)                           # (사용자, K) 적중 여부
    hits, dcg = hit.sum(1), (hit * DISC).sum(1)
    ideal = lambda n: DISC[:min(K, int(n))].sum()                 # 이상적 DCG: 1 을 min(K, 정답 수)개
    idcg_all = np.array([ideal(n) for n in rel_total])
    idcg_warm = np.array([ideal(n) for n in rel_warm])

    ok_all, ok_warm = rel_total > 0, rel_warm > 0                 # 정답 0 인 사용자는 분모에서 제외
    valid = np.isfinite(np.take_along_axis(S, top, 1))            # 유효 추천(-inf 아님)
    full_k = valid.sum(1) == K
    m = dict(
        eval_users=len(users), users_without_relevant=int((~ok_all).sum()), ranked_users=int(ok_all.sum()),
        relevant_total=int(rel_total.sum()), relevant_warm=int(rel_warm.sum()),
        relevant_cold=int(rel_total.sum() - rel_warm.sum()),
        warm_ranked_users=int(ok_warm.sum()), candidates=len(cand),
        candidates_per_user_mean=float(np.isfinite(S).sum(1).mean()), train_pairs=len(d["train"]),
        eval_pairs=len(ev),
        recall_at_10=float((hits[ok_all] / rel_total[ok_all]).mean()),          # 전체 (cold 정답은 miss)
        ndcg_at_10=float((dcg[ok_all] / idcg_all[ok_all]).mean()),
        recall_at_10_warm=float((hits[ok_warm] / rel_warm[ok_warm]).mean()),    # warm-only
        ndcg_at_10_warm=float((dcg[ok_warm] / idcg_warm[ok_warm]).mean()),
        recommendable_user_ratio=float(full_k[ok_all].mean()),
        item_coverage=float(len(np.unique(top[ok_all][valid[ok_all]])) / len(cand)),
    )

    # 평점 예측 (test 쌍 전체 대상). 못 맞힌 쌍은 global mean 으로 채운 값과 지원 쌍만의 값을 따로 보고.
    p = model.predict(ev.user_id.to_numpy(), ev.place_id.to_numpy())
    if p is None:
        m.update(mae=None, rmse=None, mae_supported=None, rmse_supported=None, rating_supported_pairs=0)
    else:
        sup = ~np.isnan(p)
        y = ev.rating.to_numpy()
        full = np.clip(np.where(sup, p, d["global_mean"]), 1, 5)
        e = full - y
        es = e[sup]
        m.update(mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
                 mae_supported=float(np.abs(es).mean()) if sup.any() else None,
                 rmse_supported=float(np.sqrt((es ** 2).mean())) if sup.any() else None,
                 rating_supported_pairs=int(sup.sum()))

    # predictions.csv: 순위 순으로 (-inf 는 추천 불가라 제외). 기본은 사용자당 상위 top_n 개.
    n = min(top_n, len(cand)) if top_n else len(cand)
    order = np.argsort(-S, axis=1, kind="stable")[:, :n]
    sc = np.take_along_axis(S, order, 1)
    keep = np.isfinite(sc)
    preds = pd.DataFrame(dict(user_id=np.repeat(users, n).reshape(len(users), n)[keep],
                              place_id=cand[order][keep], score=sc[keep]))
    preds["split"], preds["year"] = split, d["year"]
    return m, preds


# ------------------------------------------------------------------ 저장
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info():
    """Git 이 없거나 저장소가 아니면 'unknown'."""
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return dict(commit=run("rev-parse", "HEAD"), dirty=bool(run("status", "--porcelain")))
    except Exception:
        return dict(commit="unknown", dirty=None)


def save_results(d, name, split, metrics, preds, input_dir, params, seconds):
    # 연도 이동처럼 다른 실험은 d["protocol"] 로 폴더를 분리함 (연도 내 실험과 섞지 않기 위해)
    out = ROOT / "outputs/experiments" / d.get("protocol", f"known_user_{d['year']}") / name / split
    out.mkdir(parents=True, exist_ok=True)
    config = dict(model=name, params=params, year=d["year"], split=split, experiment=CFG)
    cfg_text = json.dumps(config, sort_keys=True, ensure_ascii=False)
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = dict(model=name, year=d["year"], split=split, seed=CFG["seed"], git=git_info(),
                config_sha256=hashlib.sha256(cfg_text.encode("utf-8")).hexdigest(),
                data_sha256={f: sha256(input_dir / f) for f in ("interactions.csv", "places.csv", "users.csv")},
                python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                seconds=round(seconds, 2))
    (out / "metrics.json").write_text(json.dumps(dict(meta=meta, metrics=metrics), indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    preds.to_csv(out / "predictions.csv", index=False, encoding="utf-8")
    (out / "environment.txt").write_text(f"python {sys.version}\nplatform {platform.platform()}\n"
                                         f"pandas {pd.__version__}\nnumpy {np.__version__}\n", encoding="utf-8")
    return out


def show(name, split, m):
    f = lambda v: "   n/a " if v is None else f"{v:7.4f}"
    print(f"\n[{name} | {split}] eval users {m['eval_users']}, ranked {m['ranked_users']} "
          f"(no relevant: {m['users_without_relevant']}), relevant {m['relevant_total']} "
          f"(warm {m['relevant_warm']}, cold {m['relevant_cold']}), candidates {m['candidates']}")
    print(f"  overall  Recall@10 {f(m['recall_at_10'])}  NDCG@10 {f(m['ndcg_at_10'])}")
    print(f"  warm     Recall@10 {f(m['recall_at_10_warm'])}  NDCG@10 {f(m['ndcg_at_10_warm'])}")
    print(f"  coverage recommendable users {f(m['recommendable_user_ratio'])}  item coverage {f(m['item_coverage'])}")
    print(f"  rating   MAE {f(m['mae'])}  RMSE {f(m['rmse'])}  (supported pairs {m['rating_supported_pairs']}/{m['eval_pairs']}: "
          f"MAE {f(m['mae_supported'])}  RMSE {f(m['rmse_supported'])})")


# ------------------------------------------------------------------ 자체 점검
class Oracle:
    """정답 장소에만 1점을 주는 가짜 모델. 평가기가 맞다면 warm 지표가 정확히 1.0 이 되어야 함."""

    def __init__(self, ev):
        self.rel = ev[ev.rating >= REL]

    def score(self, users, cand):
        S = np.zeros((len(users), len(cand)))
        u = pd.Series(np.arange(len(users)), index=users)
        c = pd.Series(np.arange(len(cand)), index=cand)
        r = self.rel[self.rel.place_id.isin(set(cand)) & self.rel.user_id.isin(set(users))]
        S[u[r.user_id].to_numpy(), c[r.place_id].to_numpy()] = 1.0
        return S

    def predict(self, users, places):
        return None


def self_check(d, split="validation"):
    ev = d[split]
    # 1) 오라클: warm 지표 1.0, 전체 Recall = warm 정답 비율
    m, _ = evaluate(d, Oracle(ev), split)
    assert abs(m["recall_at_10_warm"] - 1) < 1e-12 and abs(m["ndcg_at_10_warm"] - 1) < 1e-12, "oracle warm != 1"
    # 전체 Recall 은 (warm 정답 / 전체 정답) 과 같아야 함. 사용자당 정답이 1개인 분할(holdout)에서만 정확히 성립.
    assert abs(m["recall_at_10"] - m["relevant_warm"] / m["relevant_total"]) < 1e-9, "oracle overall mismatch"
    print(f"  oracle: warm Recall/NDCG = 1.0, overall Recall {m['recall_at_10']:.4f} "
          f"(warm {m['relevant_warm']} / relevant {m['relevant_total']} = {m['relevant_warm'] / m['relevant_total']:.4f})")

    # 2) 동점 규칙: global_mean 의 상위 10 개 = (후보 - 자기 train 장소) 의 place_id 사전순 앞 10 개
    g = Baseline("global_mean").fit(d)
    m, preds = evaluate(d, g, split, top_n=K)
    seen = d["train"].groupby("user_id").place_id.apply(set)
    got = preds.groupby("user_id", sort=False).place_id.apply(list)
    for u in list(got.index)[:200]:
        want = [c for c in d["cand"] if c not in seen[u]][:K]
        assert got[u] == want, f"tie order mismatch for {u}"
    print("  tie order: first 200 users match place_id lexicographic order")

    # 3) 순수 파이썬으로 지표를 다시 계산해 벡터 계산과 비교 (popularity, 상위 10)
    pop = Baseline("popularity").fit(d)
    m, preds = evaluate(d, pop, split, top_n=K)
    relset = ev[ev.rating >= REL].groupby("user_id").place_id.apply(set)
    rec, ndcg = [], []
    for u, lst in preds.groupby("user_id", sort=False).place_id.apply(list).items():
        if u not in relset.index:
            continue
        r = relset[u]
        assert not (set(lst) & set(seen[u])), "train place recommended"
        h = [1 if p in r else 0 for p in lst]
        rec.append(sum(h) / len(r))
        ndcg.append(sum(x / np.log2(i + 2) for i, x in enumerate(h)) / sum(1 / np.log2(i + 2) for i in range(min(K, len(r)))))
    assert abs(np.mean(rec) - m["recall_at_10"]) < 1e-9 and abs(np.mean(ndcg) - m["ndcg_at_10"]) < 1e-9, "metric mismatch"
    print(f"  recompute: popularity Recall {np.mean(rec):.4f}, NDCG {np.mean(ndcg):.4f} match vectorized values; "
          f"no train place recommended")
    print("SELF-CHECK PASS")


# ------------------------------------------------------------------ 실행
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=ROOT / "outputs/processed")
    ap.add_argument("--year", default=CFG["primary_year"])
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--models", nargs="+", default=CFG["models"]["baselines"])
    ap.add_argument("--predictions-top", type=int, default=100, help="predictions.csv 에 사용자당 저장할 개수 (0 = 후보 전체)")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    d = load_data(args.year, args.input_dir)
    print(f"year {d['year']}: train pairs {len(d['train']):,}, validation pairs {len(d['validation']):,}, "
          f"test pairs {len(d['test']):,}, candidates {len(d['cand']):,}, global mean {d['global_mean']:.4f}")
    if args.self_check:
        self_check(d)
        return
    for name in args.models:
        t0 = time.perf_counter()
        m, preds = evaluate(d, Baseline(name).fit(d), args.split, args.predictions_top)
        out = save_results(d, name, args.split, m, preds, args.input_dir, dict(kind=name, top_n=args.predictions_top),
                           time.perf_counter() - t0)
        show(name, args.split, m)
        print(f"  saved: {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
