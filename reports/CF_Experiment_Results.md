# DateMate 협업 필터링(CF) 및 Baseline 실험 결과

> 이 문서는 B 담당자의 기존 평가 결과를 보존한 기록이다. 후보는 train 장소만 사용하며 전체 Recall/NDCG는 cold 정답을 miss로 센 값이다. v1.1의 all-item 결과가 아니다. 공통 평가 v1.1 통합·재실험은 [통합 확인](CF_Integration_v1_1.md)을 참고한다.

> **B파트 실험 결과:** 공통 전처리 데이터와 `known_user_split` 평가 프로토콜을 사용하여 Item-based KNN 협업 필터링을 구현하고, 인기순·평균 기반 Baseline과 비교하였다. Validation에서 이웃 수 `k=40`을 선택하였으며, 2022년 Test 기준 CF의 Recall@10은 0.0131, NDCG@10은 0.0061로 인기순 Baseline보다 낮았다.

## 판정

**협업 필터링 모델 구현 및 공통 평가 프로토콜 기반 성능 비교 완료.**

Item-based KNN 모델은 팀 공통 데이터셋과 평가 코드에서 정상적으로 실행되었으며, Validation 성능에 따라 최적 이웃 수를 선택하였다.

다만 Test에서 인기순 추천이 CF보다 높은 랭킹 성능을 보였다. 이는 사용자–장소 상호작용의 높은 희소성과 Cold Item 비율이 영향을 주었을 가능성이 있다. 따라서 CF 단독 모델의 우수성을 주장하기보다는, 이후 Hybrid 모델에서 활용 가능한 개인화 추천 구성요소로 평가한다.

## 1. 구현 내용

### 1.1 협업 필터링 모델

`scripts/cf.py`에 Item-based KNN Collaborative Filtering을 구현하였다.

- 사용자–장소 만족도 평점을 이용하여 희소 행렬을 구성한다.
- 장소별 사용자 평점 벡터 간 Cosine Similarity를 계산한다.
- 유사한 장소들의 평점과 유사도 가중 평균을 이용해 추천 점수를 산출한다.
- `k=10, 20, 40`을 비교하여 이웃 수에 따른 성능 변화를 확인한다.
- `score(users, cand)`와 `predict(users, places)`를 제공하여 팀 공통 `eval_protocol.py`와 연동한다.

추천 점수를 계산할 수 없는 사용자–장소 조합에는 `-inf`를, 개별 평점 예측이 불가능한 경우에는 `NaN`을 반환하도록 구현하였다.

### 1.2 Baseline 모델

팀 공통 평가 코드인 `scripts/eval_protocol.py`에 구현된 다음 Baseline을 재사용하였다.

| 모델 | 설명 |
|---|---|
| Global Mean | 전체 학습 평점의 평균을 기준으로 예측 |
| Item Mean | 장소별 학습 평점의 평균을 기준으로 예측 |
| Popularity | 학습 데이터에서 방문 빈도가 높은 장소를 우선 추천 |

Baseline 코드를 별도로 중복 구현하지 않고 동일 평가 환경에서 CF와 비교하였다.

## 2. 데이터 및 평가 설계

모델 학습과 평가에는 `scripts/build_common_dataset.py`로 생성한 공통 전처리 데이터를 사용하였다.

입력 파일은 `outputs/processed/interactions.csv`, `places.csv`, `users.csv`이며, 원본 Training/Validation 구분이 아닌 공통 데이터셋의 `known_user_split`을 사용하였다.

| 구분 | 사용 데이터 | 목적 |
|---|---|---|
| Train | `train` + `train_only` | 모델 학습 |
| Validation | `validation` | K값 선택 |
| Test | `test` | 최종 모델 비교 |

2022년 평가 데이터는 학습 상호작용 12,167개, Validation 1,243개, Test 1,243개로 구성되었다. 추천 후보는 학습에 등장한 장소 7,172개로 제한하였다.

사용자가 학습 데이터에서 이미 평가한 장소는 추천 순위 계산 시 제외하였다. 관련 장소 판정과 Recall@10, NDCG@10, MAE, RMSE, Coverage 계산에는 팀 공통 평가 규칙을 적용하였다.

## 3. Validation: K값 선택

Item-based KNN의 이웃 수를 10, 20, 40으로 변경하여 Validation 성능을 비교하였다.

| K | Recall@10 | NDCG@10 | Warm Recall@10 | Warm NDCG@10 | MAE | RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.0113 | 0.0047 | 0.0215 | 0.0089 | 0.6423 | 0.7795 |
| 20 | 0.0159 | 0.0067 | 0.0304 | 0.0129 | 0.6423 | 0.7892 |
| **40** | **0.0216** | **0.0098** | **0.0411** | **0.0188** | **0.6388** | 0.7962 |

K값이 증가하면서 Recall@10과 NDCG@10이 개선되는 경향을 보였다.

추천 순위의 품질을 반영하는 **Validation NDCG@10을 우선 선택 기준으로 설정하여 `k=40`을 최종 모델로 선정하였다.**

단, K값이 커질수록 RMSE도 증가하므로 평점 예측 오차와 추천 랭킹 성능이 동일한 방향으로 개선되지는 않았다.

## 4. Test: CF와 Baseline 비교

Validation에서 선정한 Item-KNN(`k=40`)과 공통 Baseline의 2022년 Test 결과를 비교하였다.

| 모델 | Recall@10 | NDCG@10 | Warm Recall@10 | Warm NDCG@10 | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Global Mean | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6643 | 0.8026 |
| Item Mean | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7010 | 0.8806 |
| **Popularity** | **0.0402** | **0.0211** | **0.0754** | **0.0396** | 해당 없음 | 해당 없음 |
| Item-KNN (`k=40`) | 0.0131 | 0.0061 | 0.0246 | 0.0114 | 0.6633 | 0.8223 |

### 4.1 추천 랭킹 성능

Popularity Baseline은 Recall@10 0.0402, NDCG@10 0.0211로 가장 높은 랭킹 성능을 보였다.

Item-KNN의 Recall@10은 0.0131, NDCG@10은 0.0061로, 단독 CF가 인기순 추천보다 높은 성능을 보이지는 못하였다.

다만 Item-KNN은 전역적으로 동일한 인기 장소 순위가 아닌 사용자별 방문 이력에 기반한 점수를 생성한다는 점에서 개인화 추천 구성요소로 활용할 수 있다.

### 4.2 Warm Item 성능

Test 평가 사용자 1,243명 중 관련 정답이 있는 사용자는 1,069명이었다.

관련 정답 1,069개 중 Warm Item은 570개, Cold Item은 499개였다.

Item-KNN의 Warm Recall@10은 0.0246으로 전체 Recall@10 0.0131보다 높았다. 이는 학습 이력이 있는 장소에서 CF의 추천 성능이 상대적으로 높게 나타남을 보여준다.

### 4.3 평점 예측 성능

Item-KNN의 MAE는 0.6633으로 Global Mean의 0.6643보다 소폭 낮았다.

반면 RMSE는 Item-KNN이 0.8223으로 Global Mean의 0.8026보다 높았다. 따라서 전체적인 평점 예측 성능이 Baseline보다 개선되었다고 단정하기 어렵다.

## 5. 결과 해석 및 한계

### 5.1 사용자–장소 행렬의 희소성

2022년 공통 전처리 데이터의 사용자–장소 상호작용 행렬 희소도는 약 99.944%이다.

대부분의 사용자가 일부 장소만 평가하였기 때문에, 장소 간 공통 평가 사용자가 부족할 수 있다. 이 경우 Cosine Similarity를 이용한 이웃 관계가 충분히 형성되지 않아 CF 성능이 제한될 가능성이 있다.

### 5.2 Cold Item 문제

Test 관련 정답 중 499개는 학습 데이터에서 관측되지 않은 Cold Item이었다.

현재 Item-KNN은 학습에 등장한 장소를 기준으로 유사도를 구성하므로 학습 이력이 없는 장소에는 직접적인 개인화 점수를 부여하기 어렵다.

이 문제는 장소 이름·지역 등 속성을 이용하는 Content-based Filtering과의 결합을 통해 보완할 수 있다.

### 5.3 평가 지표 해석

`recommendable_user_ratio`는 팀 평가기에서 유효한 Top-10 추천 결과를 제공할 수 있는 사용자 비율을 나타낸다. 이는 전체 사용자–장소 조합의 점수 예측 성공률과는 다르다.

평점 예측이 불가능한 경우 공통 평가 코드의 전체 평균 Fallback이 적용될 수 있으므로, 전체 MAE/RMSE와 모델이 직접 예측한 조합의 오차를 구분하여 해석해야 한다.

### 5.4 Hybrid 모델 연계

향후 Hybrid Filtering에서는 CF 점수와 CBF 점수의 크기 차이를 고려하여 정규화하고, CF 점수를 계산할 수 없는 경우 CBF 점수로 대체하는 방식을 검토할 수 있다.

두 모델이 동일한 사용자 및 장소 후보 순서를 사용하도록 맞추고, 점수 결합 방식은 C파트 담당자와 협의해야 한다.

## 6. 파일과 재현

프로젝트 루트에서 다음 명령을 실행한다.

사전에 공통 전처리를 수행하여 `outputs/processed/` 아래에 필요한 CSV 파일을 생성해야 한다. 원본 데이터와 공통 전처리 결과 파일은 GitHub 업로드 대상에서 제외한다.

### 6.1 Validation 튜닝

```powershell
python scripts\cf.py --year 2022 --tune
```

이 명령은 `k=10, 20, 40`을 Validation에서 비교하고 최적 K값을 출력한다.

주요 튜닝 결과는 다음 위치에 저장된다.

`outputs/experiments/known_user_2022/item_knn/tuning_validation.csv`

### 6.2 최종 Test 재현

```powershell
python scripts\cf.py --year 2022 --k 40 --split test
```

K값은 Validation 성능을 기준으로 선택한 값이며, Test 결과를 이용해 재선정하지 않았다.

### 6.3 Baseline 재현

```powershell
python scripts\eval_protocol.py --year 2022 --split test
```

모델별 평가 결과는 `outputs/experiments/known_user_2022/` 아래에 저장된다. CF 결과에는 `metrics.json`, `predictions.csv`, `config.json` 등의 파일이 포함된다.

GitHub에 공유할 주요 파일은 다음과 같다.

- `scripts/cf.py`: 협업 필터링 모델 구현 및 실행 코드
- `reports/B_CF_실험결과.md`: 실험 방법, 성능 및 한계 정리
- `reports/validation_k_comparison.csv`: Validation K값 비교
- `reports/test_model_comparison.csv`: Test 모델별 성능 비교

위 성능 수치는 담당자의 로컬 환경에서 실행한 평가 결과에 기반한다. 팀 공통 데이터셋이나 평가 프로토콜이 변경되면 재현 실험 및 결과 갱신이 필요하다.

## 팀 공유용 요약

DateMate B파트에서 Item-based KNN Collaborative Filtering을 구현하고, 공통 전처리 데이터와 평가 프로토콜을 사용하여 2022년 Known-user 추천 성능을 평가하였다. Validation에서 K값 10, 20, 40을 비교한 결과 NDCG@10이 가장 높은 `k=40`을 선택하였다.

최종 Test에서 CF의 Recall@10은 0.0131, NDCG@10은 0.0061로, 인기순 Baseline의 Recall@10 0.0402, NDCG@10 0.0211보다 낮았다. 사용자–장소 행렬의 높은 희소성과 Cold Item 비율이 성능을 제약했을 가능성이 있다.

따라서 CF 단독 모델의 성능 우위를 확인하지는 못했으나, 사용자별 방문 이력에 기반한 개인화 점수 산출 기능을 구현하였으며, 이후 CBF와 결합하는 Hybrid 추천 모델의 입력으로 활용할 수 있다.
