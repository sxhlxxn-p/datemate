"""공통 데이터(outputs/processed)를 CBF 실험 관점에서 요약함. 모델 입력이 아니라 확인용임.

docs/evaluation.md 의 규칙대로 주 실험(known_user_split)의 평가 사용자 수, 후보 수, warm 비율, 정답 수를 셈.
수치는 PR 의 "평가 사용자 수와 후보 수" 항목에 그대로 옮겨 적음.

사용: python scripts/summarize_protocol.py [--input-dir outputs/processed]
필요 패키지: pandas (requirements.txt 에 이미 있음)
"""
from pathlib import Path
import argparse
import pandas as pd

root = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--input-dir", type=Path, default=root / "outputs/processed")
args = ap.parse_args()

inter = pd.read_csv(args.input_dir / "interactions.csv", dtype={"year": str})
places = pd.read_csv(args.input_dir / "places.csv")
users = pd.read_csv(args.input_dir / "users.csv", dtype={"year": str})


def title(s):
    print("\n" + "=" * 78 + f"\n {s}\n" + "=" * 78)


def pct(a, b):
    return f"{a / b * 100:.1f}%" if b else "-"


# ---------------------------------------------------------------- A. interactions 개요
title("A. interactions.csv 개요 (한 행 = 한 번의 방문 평가)")
for y, g in inter.groupby("year"):
    per_user = g.groupby("user_id")["place_id"].nunique()
    print(f"[{y}] 행 {len(g):,} | 사용자 {g.user_id.nunique():,} | 장소 {g.place_id.nunique():,} | 사용자-장소 쌍 {g[['user_id', 'place_id']].drop_duplicates().shape[0]:,}")
    print(f"     사용자당 고유 장소: 중앙값 {per_user.median():.0f}, 평균 {per_user.mean():.1f}, 최대 {per_user.max()} | 5곳 이상 {pct((per_user >= 5).sum(), len(per_user))}")
    print(f"     평점 분포: {g.rating.value_counts().sort_index().to_dict()}  (4점 이상 {pct((g.rating >= 4).sum(), len(g))})")
    print(f"     방문 유형 코드: {g.visit_type_code.value_counts().sort_index().to_dict()}")
print("\n결측률(%):", (inter.isna().mean() * 100).round(1)[lambda s: s > 0].to_dict())

# 상호작용에 들어 있는 장소가 모두 '추천 후보'인가?
cand = places.set_index("place_id")["is_model_candidate"]
inter["is_candidate_place"] = inter["place_id"].map(cand)
print("\n상호작용 장소 중 후보(is_model_candidate=1) 비율:", pct((inter.is_candidate_place == 1).sum(), len(inter)),
      "| 후보가 아닌 장소의 상호작용:", int((inter.is_candidate_place != 1).sum()), "행")

# ---------------------------------------------------------------- B. 주 실험 구조
title("B. 평가 구조 (docs/evaluation.md 규칙): known_user_split")
for y in ("2022", "2023"):
    g = inter[inter.year == y]
    print(f"\n[{y}] 분할별 행: {g.known_user_split.value_counts().to_dict()}")
    train = g[g.known_user_split.isin(["train", "train_only"])]
    test = g[g.known_user_split == "test"]
    val = g[g.known_user_split == "validation"]
    # 같은 사용자-장소는 같은 분할에만 있고, 분할 안에서 평균을 냄 (pair_aggregation = mean_within_split)
    test_pair = test.groupby(["user_id", "place_id"], as_index=False)["rating"].mean()
    val_pair = val.groupby(["user_id", "place_id"], as_index=False)["rating"].mean()
    train_pair = train.groupby(["user_id", "place_id"], as_index=False)["rating"].mean()
    n_users_test = test_pair.user_id.nunique()
    print(f"     평가 사용자(test 가 있는 사용자): {n_users_test:,} | test 쌍 {len(test_pair):,} (사용자당 {len(test_pair) / n_users_test:.2f}) | validation 사용자 {val_pair.user_id.nunique():,}")
    train_places = set(train.place_id)
    print(f"     후보 장소(같은 해 train/train_only 의 장소): {len(train_places):,}곳 | train 쌍 {len(train_pair):,}")
    upu = train_pair.groupby("user_id").size()
    print(f"     평가 사용자의 train 장소 수: 중앙값 {upu.reindex(test_pair.user_id.unique()).median():.0f}, 평균 {upu.reindex(test_pair.user_id.unique()).mean():.1f}")
    # warm: test 장소가 train 에 존재하는가
    test_pair["warm"] = test_pair.place_id.isin(train_places)
    test_pair["relevant"] = test_pair.rating >= 4
    rel = test_pair[test_pair.relevant]
    print(f"     정답(평점>=4) test 쌍: {len(rel):,} / {len(test_pair):,} ({pct(len(rel), len(test_pair))}) | 정답이 하나도 없는 평가 사용자: {n_users_test - rel.user_id.nunique():,}명 (랭킹 분모에서 제외)")
    print(f"     정답 중 warm(후보 안에 존재): {int(rel.warm.sum()):,} / {len(rel):,} ({pct(int(rel.warm.sum()), len(rel))})  <- 전체 Recall@10 의 이론적 상한 근처")
    users_with_warm = rel[rel.warm].user_id.nunique()
    print(f"     warm 정답이 1개 이상인 사용자: {users_with_warm:,} / {rel.user_id.nunique():,} ({pct(users_with_warm, rel.user_id.nunique())})")
    flag = g.drop_duplicates(["user_id", "place_id", "known_user_split"])
    flag = flag[flag.known_user_split == "test"]
    print(f"     데이터의 is_warm_item 열(test 행): {pct(int(flag.is_warm_item_in_known_user_train.sum()), len(flag))}  (내가 계산한 test 쌍 warm: {pct(int(test_pair.warm.sum()), len(test_pair))})")
    # 후보 수: 사용자마다 후보에서 자기 train 장소를 뺌
    print(f"     사용자별 후보 수: 약 {len(train_places) - upu.reindex(test_pair.user_id.unique()).mean():,.0f}곳 (후보 {len(train_places):,} - 자기 train 장소 평균)")

# ---------------------------------------------------------------- C. 연도 이동
title("C. 연도 이동 (temporal_split): 2022 학습 -> 2023 평가")
tr22 = set(inter[(inter.year == "2022")].place_id)
g23 = inter[inter.year == "2023"]
pair23 = g23.groupby(["user_id", "place_id"], as_index=False)["rating"].mean()
rel23 = pair23[pair23.rating >= 4]
print(f"2022 장소 {len(tr22):,}곳 | 2023 장소 {g23.place_id.nunique():,}곳 | 공통 {len(tr22 & set(g23.place_id)):,}곳 (명세의 공통 모델 장소 1,609)")
print(f"2023 정답(평점>=4) 쌍 {len(rel23):,} 중 2022 에 있는 장소: {pct(rel23.place_id.isin(tr22).sum(), len(rel23))}")
print(f"2023 사용자 {g23.user_id.nunique():,}명은 모두 cold user (2023 평점을 취향에 쓰면 안 됨)")

# ---------------------------------------------------------------- D. places
title("D. places.csv (한 행 = 장소 하나)")
c = places[places.is_model_candidate == 1]
print(f"전체 {len(places):,}곳 | 추천 후보 {len(c):,}곳")
print("후보 중 결측률(%):", (c.isna().mean() * 100).round(1)[lambda s: s > 0].to_dict())
print("후보의 방문 유형 코드:", c.visit_type_code.value_counts().sort_index().to_dict())
print("장소를 구분한 방법(identity_method):", c.identity_method.value_counts().to_dict())
print("신뢰도(identity_confidence):", c.identity_confidence.value_counts().to_dict())
print("POI 대분류 있는 후보:", pct(c.poi_category_large.notna().sum(), len(c)), "| 좌표 있는 후보:", pct(c.latitude.notna().sum(), len(c)),
      "| 도로명주소 있는 후보:", pct(c.road_address.notna().sum(), len(c)), "| 지번주소 있는 후보:", pct(c.lot_address.notna().sum(), len(c)))
nm = c.place_name.fillna("").astype(str)
print(f"장소 이름: 같은 이름이 여러 장소(place_id)인 이름 수 {int((c.groupby('place_name').size() > 1).sum()):,} | 이름 길이 중앙값 {nm.str.len().median():.0f}자 | 공백 없는 이름 {pct(int((~nm.str.contains(' ')).sum()), len(nm))}")
print("후보 장소의 연도:", c.source_years.value_counts().to_dict())

# ---------------------------------------------------------------- E. users
title("E. users.csv (한 행 = 사용자 하나)")
print(f"사용자 {len(users):,}명 | 연도별 {users.year.value_counts().to_dict()}")
print("결측률(%):", (users.isna().mean() * 100).round(1)[lambda s: s > 0].to_dict())
print("성별:", users.gender.value_counts(dropna=False).to_dict())
print("연령대:", users.age_group.value_counts(dropna=False).sort_index().to_dict())
print("선호 시도 1순위:", users.preferred_sido_1.value_counts().head(6).to_dict())
for k in (1, 2, 3, 4):
    print(f"travel_style_{k}:", users[f"travel_style_{k}"].value_counts().sort_index().to_dict())
print("상호작용의 사용자가 users.csv 에 있는 비율:", pct(inter.user_id.isin(users.user_id).sum(), len(inter)), "(행 기준)")
