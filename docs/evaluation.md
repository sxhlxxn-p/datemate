# 공통 실험 및 평가 명세 v1.0

이 문서는 모델 구현 전 팀 공통 실험 계약이다. 설정은 `configs/experiment.json`으로 고정한다. 기존 `data_spec.md`의 키·분할과 누수 방지 기준을 변경하지 않는다. 모델 성능 결과가 아니다.

## 실험 구분

| 실험 | 학습 | 튜닝 | 최종 평가 | 해석 |
|---|---|---|---|---|
| 주 실험 | 2022 known_user_split=train/train_only | 같은 연도 validation | 같은 연도 test | 기존 사용자의 새 장소 |
| 보조 연도 내 실험 | 2023 train/train_only | 2023 validation | 2023 test | 독립된 별도 모델 |
| 연도 이동 | 2022 전체 또는 개발 분할에서 확정한 학습 절차 | 2022 내부 validation에서만 결정 | 2023 전체 | 모든 사용자가 cold user |

2023 test를 보면서 선택한 파라미터는 연도 이동 실험에 사용하지 않는다. 연도 내 2023 실험과 연도 이동 실험의 모델·출력 경로를 분리한다. 연도 이동의 신규 사용자는 서비스에서 사전 입력 가능한 프로필만 사용하며 2023 평가 평점을 사용자 벡터에 넣지 않는다. 원본 source_split은 출처다.

## 집계와 누수

동일 사용자–장소는 같은 분할에만 있다. 분할 후 각 분할 안에서만 반복 평점 평균을 구한다. train_only는 학습에 포함하지만 평가 사용자 분모에서는 제외한다. 고유 장소 5개 이상인 사용자의 마지막 장소는 test, 직전 장소는 validation이다. 정렬은 해당 사용자–장소의 **마지막 방문** 날짜·순서·안정적인 해시 기준이다. 반복 방문의 이전 날짜까지 해당 쌍 분할로 이동하므로 전역 시점 기준의 엄격한 시간 분할을 보장하지 않는다. 이 분할을 사용자별 장소 holdout이라고 보고한다.

TF-IDF, 정규화, 평균, 인기, 이웃, 가중치 보정과 후보 선별 통계는 해당 train에서만 계산한다. validation은 튜닝용이고 test는 최종 1회 비교용이다. validation을 train에 다시 넣는 후속 실험은 별도 버전으로 수행하며 주 실험과 섞지 않는다.

다음 열은 전체 기간 또는 결과값을 포함하므로 모델 입력 금지다.

- interactions: rating, revisit_intention, recommendation_intention. rating은 학습 목표와 train 사용자 이력 계산에만 사용.
- places: mean_rating, visit_count, rated_visit_count, unique_user_count, first_visit_date, last_visit_date, source_years.
- users: interaction_count.
- user_id, travel_id, interaction_id, 원본 분할은 키/감사용이며 내용 특성이 아니다. CF의 user/item 인덱스는 허용한다.

**카탈로그 한계:** 기존 장소 통합과 대표 이름·주소·방문 유형은 두 연도 전체 방문에서 계산한다. 따라서 현재 산출물은 전체 기간 카탈로그를 이용한 탐색적 평가다. 미래 시점까지 완전히 차단된 실험이라고 주장할 수 없다. train에 등장한 장소만 후보로 제한해도 이 한계가 완전히 없어지지는 않는다. 엄격한 inductive/시간 이동 주장에는 train 시점 장소 통합 및 당시 이용 가능한 독립 카탈로그를 다시 구성하는 별도 전처리가 필요하다. 현재 스키마·기준 수치를 보존하기 위해 여기서는 그 재구성을 수행하지 않는다.

## 동일 조건의 지표

1. 평점 예측: test 사용자–장소 평균 평점에 MAE/RMSE. 예측값 범위는 [1,5]로 고정. 모델별 같은 평가 쌍을 사용하고 fallback을 포함한 전체값 및 지원 가능한 쌍만의 값을 분리한다.
2. 주 랭킹 후보: 해당 연도 train/train_only에 존재하는 후보 장소 전체에서 사용자 train 장소를 제외. 음성 샘플링은 하지 않는다. 미관측은 관측 부재이지 싫어함 라벨이 아니다.
3. 정답: 해당 split의 사용자–장소 평균 rating>=4. 이진 gain으로 NDCG@10/Recall@10을 사용자별 계산 후 macro 평균. 동점은 place_id 사전순으로 고정. 이상적인 DCG는 min(10, relevant_count)개 1을 배치한 값이다.
4. 전체 정답이 0인 사용자는 랭킹 분모에서 제외하되 제외 수를 보고한다. 후보에 없는 cold 정답은 전체 지표에서 miss(0)이며 warm-only 지표에서는 제거한다. 전체/후보 내 정답 수, 평가 사용자 수를 반드시 기록한다.
5. CF 주 지표는 같은 연도 train에 존재하는 warm-item 정답. CBF/Hybrid도 이 동일 하위집합 지표를 제공하고 전체 정답 지표를 추가한다. train 후보 제한 아래에서는 cold 정답을 추천할 수 없다는 점을 명시한다. 더 넓은 카탈로그를 쓰는 CBF cold-item 실험은 별도 표에 기록한다.
6. 추천 가능 사용자 비율 = K개 유효 추천을 제공한 사용자 / 정답이 있는 평가 사용자. 실제 점수를 못 낸 사용자도 전체 분모에 포함한다. item coverage = 추천된 고유 장소 / 공통 후보 장소. 후보가 K개 미만이면 가용 개수를 보고한다.
7. 연도 이동은 2022 train 카탈로그 후보 및 2022 학습 장소와 2023 정답의 공통/신규 장소를 구분한다. 순수 CF cold-user 결과는 개인화 성능으로 해석하지 않는다. 전체 1,609개 공통 모델 장소라는 기존 수치는 전기간 집계이며 train 공통 장소 수와 다를 수 있다.

기준선은 global mean, train item mean(미지원 시 global mean), train 사용자–장소 쌍 빈도 기반 popularity다. 후보와 전처리를 바꾼 비교는 별도 실험이다. seed=42, 단일 실행 결과와 여러 seed 평균을 혼동하지 않는다.

## 결과 인터페이스

로컬 `outputs/experiments/<protocol>/<model>/`에 config.json, metrics.json, predictions.csv, environment.txt를 보관한다. predictions.csv는 user_id, place_id, score, split, year이며 순위 전체 후보 점수를 기록한다. 행 단위 예측·프로필·학습 모델은 Git에 올리지 않는다.

공유 결과 JSON은 모델명, Git commit, 설정 해시, 로컬 데이터 산출물 SHA-256, Python/패키지 버전, seed, 연도/분할, train 쌍 수, 평가 쌍·사용자 수, warm/cold 분모, 후보 수, MAE/RMSE/NDCG@10/Recall@10/coverage, 실행 시간의 집계만 포함한다. 아직 결과가 없을 때 수치를 채우지 않는다. 구현자는 이 계약에 맞는 evaluator를 함께 제출하며 기준선과 동일 후보·분모인지 검토받는다.
