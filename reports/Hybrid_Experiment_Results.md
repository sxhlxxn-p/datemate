# Hybrid 추천 구현 및 평가 결과

## 구현 범위

- CBF 코사인 점수와 Item-KNN CF 평점 예측을 고정 범위로 정규화한 뒤 `alpha` 가중합으로 결합한다.
- CF가 점수를 계산하지 못한 사용자-장소 쌍은 CBF 점수를 사용한다.
- `alpha`는 warm validation NDCG@10 warm으로 선택하고, 선택 후 같은 모델로 warm/all 후보를 평가한다.
- 커플 시연은 두 사용자 점수의 평균과 최솟값을 각각 계산하며, 두 사용자의 train 방문 장소 합집합을 제외한다.

## 재현 환경과 선택 결과

공통 전처리 파일과 평가 프로토콜 v1.1을 사용했다. 모델과 정규화 파라미터는 train으로 적합하고, `alpha`만 validation에서 선택했다.

| 연도 | 선택 기준 | 선택 alpha |
|---|---|---:|
| 2022 | warm validation NDCG@10 warm | 0.00 |
| 2023 | warm validation NDCG@10 warm | 0.40 |

2022의 선택값이 0.00이므로 2022 Hybrid 결과는 선택된 CBF와 동일하다. CBF보다 높은 성능으로 해석하지 않는다. 2023에서는 CF 점수가 실제 결합에 사용된다.

## 2022 최종 test

| 후보 | Recall@10 | NDCG@10 | Warm NDCG@10 |
|---|---:|---:|---:|
| warm | 0.022451 | 0.010281 | 0.019282 |
| all | 0.020580 | 0.011580 | 0.003415 |

동일한 warm test 기준 비교는 다음과 같다.

| 모델 | Recall@10 | NDCG@10 |
|---|---:|---:|
| 인기순 | 0.040225 | 0.021109 |
| CBF(region) | 0.022451 | 0.010281 |
| Hybrid, alpha=0.00 | 0.022451 | 0.010281 |
| Item-KNN CF, k=40 | 0.015903 | 0.008295 |

Hybrid는 CF보다 높지만 인기순보다 낮다. alpha=0.00이므로 Hybrid 고유의 성능 향상은 확인되지 않았다.

## 2023 validation과 최종 test

| 분할 | 후보 | Recall@10 | NDCG@10 | Warm NDCG@10 |
|---|---|---:|---:|---:|
| validation | warm | 0.051625 | 0.026404 | 0.055795 |
| validation | all | 0.049713 | 0.022258 | 0.029189 |
| test | warm | 0.025257 | 0.011514 | 0.026244 |
| test | all | 0.027128 | 0.013011 | 0.014521 |

2023은 alpha=0.40이 선택되어 CBF와 CF가 실제로 결합된다. test는 alpha를 고른 뒤 처음 확인한 held-out 결과다. validation보다 test 지표가 낮아졌으며, 2022 학습에서 2023으로 이동하는 외부 평가와는 별개의 연도 내 실험이다.

## 커플 추천 시연

2022 사용자 `2022:a000011`, `2022:a000012`와 all 후보로 실행했다. 평균 방식의 상위 결과에는 탭, 모티프, 봉주르가, 최솟값 방식에는 로얄프라자, 비손프라자, 탭이 포함됐다. 동일 장소명이 여러 `place_id`에 존재할 수 있으므로 결과 CSV에서는 장소명과 `place_id`를 함께 확인한다.

공동 만족도 정답이 없으므로 이 결과는 추천 흐름 시연이다. 실제 커플 만족도 향상이나 오프라인 정확도 근거로 사용하지 않는다.

## 실행

```sh
python scripts/run_hybrid_real.py --year 2022 --final-test
python scripts/run_hybrid_real.py --year 2023 --final-test
python scripts/run_couple_demo.py --year 2022 --candidate-mode all
```

`run_hybrid_real.py`는 요약 JSON과 함께 공통 평가 경로의 config, metrics, predictions, environment 파일을 저장한다. 원본 데이터, 개인별 전체 점수, 생성된 실험 출력은 Git에 올리지 않는다.

## 검증

- `python -m unittest discover -s tests -v`: 37개 통과
- 2022 alpha 선택, warm/all validation 및 test 재현 완료
- 2023 alpha 선택과 warm/all validation 및 test 재현 완료
- clean 커밋 `ce0d136`과 동일 데이터에서 기준선·CBF·CF·Hybrid 재실행 완료
- 실제 두 사용자 mean/min Top-10 시연 완료
