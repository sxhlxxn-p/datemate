# 실험·평가 기준 v1.1

모든 모델은 아래 분할·후보·지표를 맞춰 비교한다. 설정은 `configs/experiment.json`, 데이터 구조는 `data_spec.md`를 참고한다.

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

장소 통합과 대표 이름·주소·방문 유형은 두 연도 전체 로그에서 정했다. train 장소로 후보를 제한해도 이 과정에 평가 시점의 정보가 포함될 수 있다. 현재 실험은 전체기간 카탈로그를 쓰는 비교로 해석한다. 엄격한 시점별 평가를 하려면 train 시점의 장소 통합과 당시 이용 가능한 카탈로그를 따로 구성해야 한다.

## Warm-item / All-item

두 모드는 같은 train과 모델 파라미터를 쓰고 후보만 바꾼다. `--candidate-mode warm|all`로 실행한다.

| 모드 | 후보(사용자 train 장소 제외 전) | 정답·사용자 분모 | 주 지표 |
|---|---|---|---|
| warm | 같은 연도 train/train_only에 나온 장소 | 해당 split 평점 평균>=4이며 warm인 장소. warm 정답이 있는 사용자 전체 | `recall_at_10_warm`, `ndcg_at_10_warm` |
| all | 고정된 places.csv의 is_model_candidate=1 전체 | 해당 split 평점 평균>=4인 전체 장소. 정답이 있는 사용자 전체 | `recall_at_10`, `ndcg_at_10` |

all의 후보 목록은 현재 두 연도 로그로 만든 고정 카탈로그다. 학습에 없는 장소도 특성으로 점수를 낼 수 있는지 확인하는 실험이며, 실제로 그 시점에 새로 등록된 장소의 성능을 뜻하지 않는다. strict 시간 이동 평가는 별도 전처리가 필요하다.

warm 실행에서도 전체 정답 기준 지표를 저장하지만, 이 값은 cold 정답을 miss로 세는 진단값이다. all 실행의 지표와 섞지 않는다. all 실행의 `_warm` 값도 all 후보 사이에서 얻은 순위이므로 warm 실행의 주 지표와 다르다. **표마다 candidate_mode를 함께 적는다.**

2022 기준 분모는 아래와 같다. 후보 수는 사용자별 train 장소를 빼기 전이다.

| 항목 | validation | test |
|---|---:|---:|
| 전체 평가 사용자 | 1,243 | 1,243 |
| 평점 4점 이상 정답이 있는 사용자 | 1,066 | 1,069 |
| warm 정답이 있는 사용자 | 559 | 570 |
| cold 정답 쌍 | 507 | 499 |
| warm 후보 | 7,172 | 7,172 |
| all 후보 | 16,764 | 16,764 |

`python scripts/protocol_summary.py --year 2022`로 다시 확인한다. 모델별로 이 수가 달라지면 결과를 비교하기 전에 원인을 확인한다.

## 후보 제외·지표 계산

- 모든 모델에서 해당 사용자의 train/train_only 장소만 제외한다. validation/test 방문 여부나 평점을 보고 후보를 제거하지 않는다. validation을 재학습에 넣지 않는 현재 설정에서는 test 평가 때도 validation 방문지는 제외하지 않는다.
- 음성 샘플링 없이 전체 후보를 순위화한다. 미관측 장소를 싫어함 라벨로 사용하지 않는다.
- Top-10, binary gain으로 사용자별 Recall과 NDCG를 계산하고 macro 평균한다. 동점은 place_id 사전순. IDCG는 min(10, 해당 모드 정답 수)개 정답을 맨 위에 둔 값이다.
- 정답이 없는 사용자는 해당 지표 분모에서 제외하고 수를 기록한다. 정답이 있지만 모델이 점수를 못 낸 사용자는 제외하지 않고 0점으로 계산한다. 후보에 없거나 `-inf`인 정답도 miss다. 분모가 0이면 0점 대신 JSON null을 기록한다.
- 추천 가능 비율은 전체 정답이 있는 사용자 중 **유효 추천 10개**를 반환한 사용자 비율이다. 후보나 지원 점수가 10개 미만이면 이 비율에서는 실패로 센다. 가용 후보 수와 점수를 낼 수 있는 후보 수는 따로 기록한다.
- item coverage는 전체 정답이 있는 사용자에게 실제 추천한 고유 장소 수 / 해당 모드의 전체 후보 수다.
- 평점 지표는 모드와 관계없이 같은 split의 전체 사용자–장소 평균 평점에 MAE/RMSE를 계산한다. 예측은 [1,5]로 제한하고, 지원하지 못하는 쌍은 train 전체평균으로 채운 전체값과 지원 쌍만의 값을 구분한다. 평점 예측 기능이 없으면 null로 남긴다.

## 튜닝과 기준선

CBF·CF·Hybrid 모두 **warm 후보 validation NDCG@10 warm**으로 파라미터를 고른다. 같은 최선 파라미터로 warm/all을 평가하며, all이나 test 결과를 보고 다시 고르지 않는다. 동점은 미리 정한 grid 순서의 첫 설정을 사용한다. seed=42.

기준선은 global mean, train item mean(미지원 시 global mean), train 사용자–장소 쌍 빈도 popularity다. all 모드에서 popularity의 cold 점수는 0으로 둔다. 순수 CF가 cold 장소나 이웃 없는 장소의 점수를 못 내면 `-inf`를 반환한다. 기준선·CBF fallback을 붙인 CF는 순수 CF와 모델명을 구분한다.

2023 연도 내 실험은 같은 절차의 별도 결과표로 작성한다. 연도 이동은 기존 2022 train 후보 설정을 유지하며, 이 v1.1의 all-item 결과와 섞지 않는다.

## 공통 점수 인터페이스

모든 랭킹 모델은 `score(users, candidates)`로 전달받은 사용자·후보 순서를 바꾸지 않고 `(사용자 수, 후보 수)` 실수 배열을 반환한다. 점수를 낼 수 없는 조합만 `-inf`로 표시할 수 있으며 `NaN`과 `+inf`는 오류다. 평가기는 사용자별 train 장소를 추가로 `-inf` 처리한다.

평점 예측을 지원하는 모델은 `predict(users, places)`로 입력 길이와 같은 1차원 배열을 반환한다. 지원하지 못하는 쌍은 `NaN`, 기능 전체가 없으면 `None`을 반환한다. 평가기는 shape, NaN/무한대 규칙을 검사한 뒤 fallback 포함 전체 지표와 지원 쌍 지표를 분리한다. CF·CBF·Hybrid 모두 이 계약과 같은 후보 순서를 사용해야 공정 비교로 인정한다.

## 결과 저장

로컬 `outputs/experiments/v1_1/known_user_<year>/<warm|all>/<model>/<split>/`에 config.json, metrics.json, predictions.csv, environment.txt를 보관한다. predictions.csv는 user_id, place_id, score, split, year이며 기본 실행은 사용자별 상위 100개를 저장한다. `--predictions-top 0`일 때만 전체 후보를 저장한다. Hybrid 결합 입력이나 모델 간 후보별 점수 비교에는 반드시 전체 후보 출력(`--predictions-top 0`)을 사용한다. 행 단위 예측·프로필·학습 모델은 Git에 올리지 않는다.

공유 결과 JSON은 모델명, Git commit, 설정 해시, 로컬 데이터 산출물 SHA-256, Python과 pandas/numpy/scipy/scikit-learn 버전, seed, 연도/분할, train 쌍 수, 평가 쌍·사용자 수, warm/cold 분모, 후보 수, MAE/RMSE/NDCG@10/Recall@10/coverage, 실행 시간의 집계만 포함한다. 평가 코드도 함께 공유하고, 기준선과 후보·분모가 같은지 확인한다.


## 담당자 연결

- A(CBF): 기존 모델을 그대로 학습하고 후보 모드별 점수를 반환한다. TF-IDF/one-hot은 warm train 장소로만 fit하고 all 장소는 transform만 한다. `cbf.py`는 한 번 튜닝한 설정을 두 모드에 사용한다.
- B(CF): `load_data(year, input_dir, candidate_mode)`와 `evaluate()`를 사용한다. k 선택 시 전체 진단값이 아닌 `ndcg_at_10_warm`을 비교한다. CF CLI는 `--candidate-mode warm|all|both`를 지원하며 기본값은 both다.
- C(Hybrid): 동일한 연도·split·candidate_mode·데이터 해시·후보 해시의 점수만 연결한다. Top-100 목록의 교집합으로 후보를 줄이면 안 된다. 모델 객체의 `score()`를 직접 연결하거나 전체 출력(`--predictions-top 0`)을 사용한다. 출력에서 생략된 점수는 추천 불가이며 사용자를 삭제하는 근거가 아니다. 가중치·스케일 보정은 validation/train만 쓴다.

```python
from eval_protocol import load_data, evaluate

warm = load_data("2022", input_dir, "warm")
model = MyModel(**best_params).fit(warm)  # best_params는 warm validation에서 선택
for mode in ("warm", "all"):
    d = load_data("2022", input_dir, mode)
    metrics, predictions = evaluate(d, model, "validation", top_n=0)
```

기존 프로토콜 결과는 그대로 보관한다. v1.1 결과는 별도 경로에 저장하며 후보·분모가 다른 이전 결과를 최종 비교표에 합치지 않는다.
