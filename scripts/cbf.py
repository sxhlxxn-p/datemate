"""
Content-based Filtering (장소의 특징으로 사용자 취향과 닮은 장소를 추천).

흐름
  1. 장소 벡터: 이름 TF-IDF + 방문 유형 원-핫 (+ 선택: 주소에서 뽑은 시군구 원-핫). 모두 train 장소로만 학습.
  2. 사용자 프로필 = 사용자가 train 에서 평가한 장소 벡터의 가중합 (가중치 = 평점, 또는 평점 - 사용자 평균).
  3. 점수 = 프로필과 후보 장소 벡터의 코사인 유사도. 사용자의 train 장소는 평가기가 제외함.
  4. validation 으로 설정을 고르고, 고른 설정으로 test 는 모델당 한 번만 평가.

모델 두 가지 (결과 폴더 이름)
  cbf         : configs/experiment.json 에 정해진 특징만 (이름 TF-IDF + 방문 유형 원-핫)
  cbf_region  : 위에 주소 시군구 특징을 더한 추가 실험 (설정 파일에 없는 특징이라 따로)

사용
  python -X utf8 scripts/cbf.py --year 2022                 # validation 튜닝 + 최선 설정의 validation 결과 저장
  python -X utf8 scripts/cbf.py --year 2022 --final-test    # 위에 더해 test 를 한 번 평가 (최종)
  python -X utf8 scripts/cbf.py --cross-year                # 2022 학습 -> 2023 평가 (모든 2023 사용자는 cold user)
"""
from pathlib import Path
import argparse
import itertools
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, normalize

import eval_protocol as ep

ROOT = ep.ROOT
SIDO = {"11": "서울", "26": "부산", "27": "대구", "28": "인천", "29": "광주", "30": "대전", "31": "울산", "36": "세종",
        "41": "경기", "42": "강원", "43": "충북", "44": "충남", "45": "전북", "46": "전남", "47": "경북", "48": "경남",
        "50": "제주"}


# ------------------------------------------------------------------ 장소 정보 (모델 입력으로 허용된 열만 사용)
def sido_key(first_token):
    """주소 첫 단어 -> 두 글자 시도 이름. '충청북도' -> '충북', '경기도' -> '경기', '서울특별시' -> '서울'."""
    t = first_token
    return t[0] + t[2] if len(t) >= 4 and t[:2] in ("충청", "경상", "전라") else t[:2]


def load_place_tables(input_dir):
    """장소별 (이름, 방문 유형, 시군구, 시도) 문자열. 값이 없으면 빈 문자열.
    place_name / road_address / lot_address / visit_type_code 만 씀 (mean_rating, visit_count 등 결과값 열은 금지)."""
    p = pd.read_csv(input_dir / "places.csv", dtype=str).set_index("place_id")
    addr = p.road_address.fillna(p.lot_address).fillna("")
    toks = addr.str.split()
    sido = toks.map(lambda t: sido_key(t[0]) if t else "")
    sgg = pd.Series([f"{s} {t[1]}" if s and len(t) > 1 else "" for s, t in zip(sido, toks)], index=p.index)
    return dict(name=p.place_name.fillna(""), vtype=p.visit_type_code.fillna(""), sgg=sgg, sido=sido)


# ------------------------------------------------------------------ 모델
class CBF:
    def __init__(self, name_mode="word", w_type=1.0, w_region=0.0, weighting="raw"):
        self.name_mode, self.w_type, self.w_region, self.weighting = name_mode, w_type, w_region, weighting

    def params(self):
        return dict(name_mode=self.name_mode, w_type=self.w_type, w_region=self.w_region, weighting=self.weighting)

    def fit(self, d, tables):
        cand = d["cand"]
        self.cand, self.train = cand, d["train"]
        ids = tables["name"].index
        # 이름 TF-IDF: 어휘와 IDF 는 train 장소 이름으로만 학습하고 나머지 장소는 transform 만 함
        if self.name_mode == "word":
            tv = TfidfVectorizer(analyzer="word")
        else:  # 이름의 60% 가 공백 없는 한 단어라서 글자 2~3개 단위도 시험
            tv = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
        tv.fit(tables["name"][cand])
        blocks = [tv.transform(tables["name"])]

        def onehot(series, weight):
            # train 장소에 있던 값만 열이 됨. 빈 값/처음 보는 값은 영벡터라 유사도에 기여하지 않음.
            seen = series[cand]
            enc = OneHotEncoder(handle_unknown="ignore").fit(seen[seen != ""].to_frame())
            return weight * enc.transform(series.to_frame())

        if self.w_type:
            blocks.append(onehot(tables["vtype"], self.w_type))
        if self.w_region:
            blocks.append(onehot(tables["sgg"], self.w_region))
        self.X = normalize(sp.hstack(blocks).tocsr())             # 모든 장소 벡터 (행 길이 1)
        self.row = pd.Series(np.arange(len(ids)), index=ids)       # place_id -> X 의 행
        self.Xc = self.X[self.row[cand].to_numpy()]                # 후보 장소 벡터
        # 사용자별 train 이력 (평점 예측용)
        self.hist = {u: (self.row[g.place_id].to_numpy(), g.rating.to_numpy()) for u, g in self.train.groupby("user_id")}
        return self

    def score(self, users, cand):
        assert len(cand) == len(self.cand) and (cand == self.cand).all(), "candidate order changed"
        t = self.train[self.train.user_id.isin(users)]
        w = t.rating
        if self.weighting == "centered":                          # 평점 - 사용자 평균: '평균보다 좋았던' 곳을 강조
            c = t.rating - t.groupby("user_id").rating.transform("mean")
            allzero = c.abs().groupby(t.user_id).transform("max") == 0   # 평점이 전부 같으면 0 이 되므로 원래 평점 사용
            w = c.where(~allzero, t.rating)
        u = pd.Series(np.arange(len(users)), index=users)
        c_idx = pd.Series(np.arange(len(cand)), index=cand)
        W = sp.csr_matrix((w.to_numpy(float), (u[t.user_id].to_numpy(), c_idx[t.place_id].to_numpy())),
                          shape=(len(users), len(cand)))
        P = normalize(W @ self.Xc)                                 # 사용자 프로필 (행 길이 1)
        return (P @ self.Xc.T).toarray()                           # 코사인 유사도

    def predict(self, users, places):
        """평점 예측 = 사용자가 평가한 장소들의 평점을, 대상 장소와의 유사도로 가중평균. 유사한 이력이 없으면 NaN."""
        out = np.full(len(places), np.nan)
        rows = self.row.reindex(places).to_numpy()
        for i, (u, r) in enumerate(zip(users, rows)):
            if np.isnan(r) or u not in self.hist:
                continue
            items, ratings = self.hist[u]
            s = (self.X[int(r)] @ self.X[items].T).toarray().ravel()
            if s.sum() > 0:
                out[i] = (s * ratings).sum() / s.sum()
        return out


class ColdRegion:
    """연도 이동용 cold user 모델. 평점 이력이 없으므로 사전 입력 프로필(선호 시도 1~3순위)만 씀.
    점수 = 장소의 시도가 사용자의 선호 시도와 같으면 순위별 가중치(3,2,1). 같은 점수끼리는 place_id 사전순."""

    def __init__(self, users, sido_of_place):
        self.pref = {r.user_id: [SIDO.get(getattr(r, f"preferred_sido_{k}"), "") for k in (1, 2, 3)]
                     for r in users.itertuples()}
        self.sido = sido_of_place

    def score(self, users, cand):
        place_sido = self.sido.reindex(cand).fillna("").to_numpy()
        S = np.zeros((len(users), len(cand)))
        for i, u in enumerate(users):
            for wt, s in zip((3, 2, 1), self.pref.get(u, [])):
                if s:
                    S[i] += wt * (place_sido == s)
        return S

    def predict(self, users, places):
        return None


# ------------------------------------------------------------------ 튜닝 (validation 만 사용)
def grid_for(variant):
    regions = [0.0] if variant == "cbf" else [0.5, 1.0, 2.0]
    return list(itertools.product(["word", "char"], [0.0, 0.5, 1.0, 2.0], regions, ["raw", "centered"]))


def tune(d, tables, variant):
    rows = []
    for name_mode, w_type, w_region, weighting in grid_for(variant):
        t0 = time.perf_counter()
        m = CBF(name_mode, w_type, w_region, weighting).fit(d, tables)
        met, _ = ep.evaluate(d, m, "validation", top_n=ep.K)
        rows.append(dict(name_mode=name_mode, w_type=w_type, w_region=w_region, weighting=weighting,
                         ndcg_at_10=met["ndcg_at_10"], recall_at_10=met["recall_at_10"],
                         ndcg_at_10_warm=met["ndcg_at_10_warm"], recall_at_10_warm=met["recall_at_10_warm"],
                         mae=met["mae"], seconds=round(time.perf_counter() - t0, 1)))
    res = pd.DataFrame(rows)
    # 선택 기준: validation 전체 NDCG@10 (동점이면 표에서 먼저 나온 설정 = 더 단순한 설정)
    best = res.loc[res.ndcg_at_10.idxmax()]
    return res, dict(name_mode=best.name_mode, w_type=float(best.w_type), w_region=float(best.w_region),
                     weighting=best.weighting)


def run_year(args, tables):
    d = ep.load_data(args.year, args.input_dir)
    print(f"year {d['year']}: train pairs {len(d['train']):,}, candidates {len(d['cand']):,}")
    for variant in ("cbf", "cbf_region"):
        res, best = tune(d, tables, variant)
        folder = ROOT / "outputs/experiments" / f"known_user_{d['year']}" / variant
        folder.mkdir(parents=True, exist_ok=True)
        res.to_csv(folder / "tuning_validation.csv", index=False, encoding="utf-8")
        print(f"\n=== {variant}: {len(res)} settings tried on validation. top 5 by NDCG@10 ===")
        print(res.sort_values("ndcg_at_10", ascending=False).head(5).round(4).to_string(index=False))
        print("chosen:", best)
        splits = ["validation"] + (["test"] if args.final_test else [])
        for split in splits:
            t0 = time.perf_counter()
            m = CBF(**best).fit(d, tables)
            met, preds = ep.evaluate(d, m, split, args.predictions_top)
            ep.save_results(d, variant, split, met, preds, args.input_dir, dict(best, grid_size=len(res)),
                            time.perf_counter() - t0)
            ep.show(variant, split, met)


def run_cross_year(args, tables):
    d22 = ep.load_data("2022", args.input_dir)
    inter = pd.read_csv(args.input_dir / "interactions.csv", dtype={"year": str, "user_id": str, "place_id": str})
    g = inter[inter.year == "2023"]
    cross = g.groupby(["user_id", "place_id"], as_index=False)["rating"].mean()
    # 학습은 2022 train, 후보는 2022 train 장소. 2023 평점은 사용자 프로필에 쓰지 않음.
    d = dict(year="2023", protocol="cross_year_2022_to_2023", train=d22["train"], cross=cross, cand=d22["cand"],
             global_mean=d22["global_mean"])
    users = pd.read_csv(args.input_dir / "users.csv", dtype=str)
    users = users[users.year == "2023"]
    print(f"cross-year: 2023 users {cross.user_id.nunique():,}, pairs {len(cross):,}, candidates {len(d['cand']):,} (2022 train places)")
    models = {n: ep.Baseline(n).fit(d) for n in ("global_mean", "item_mean", "popularity")}
    models["cold_region"] = ColdRegion(users, tables["sido"])
    for name, m in models.items():
        t0 = time.perf_counter()
        met, preds = ep.evaluate(d, m, "cross", args.predictions_top)
        ep.save_results(d, name, "cross", met, preds, args.input_dir, dict(kind=name), time.perf_counter() - t0)
        ep.show(name, "cross", met)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=ROOT / "outputs/processed")
    ap.add_argument("--year", default=ep.CFG["primary_year"])
    ap.add_argument("--final-test", action="store_true", help="최선 설정으로 test 를 평가 (모델당 한 번만 실행할 것)")
    ap.add_argument("--cross-year", action="store_true")
    ap.add_argument("--predictions-top", type=int, default=100)
    args = ap.parse_args()
    tables = load_place_tables(args.input_dir)
    (run_cross_year if args.cross_year else run_year)(args, tables)


if __name__ == "__main__":
    main()
