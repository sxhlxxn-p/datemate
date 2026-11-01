# CF 공통 평가 통합 확인

2026-11-01, B파트 Item-KNN 구현을 공통 평가 v1.1에 연결했다. 기존 수치는 [원래 보고서](CF_Experiment_Results.md)와 두 비교 CSV에 보존했다. 아래는 통합 후 다시 실행한 결과다.

## 변경 내용

- warm/all 후보 모드를 지원한다. train은 같고, 한 번 학습한 모델의 후보만 바꾼다.
- warm validation NDCG@10으로 K=10,20,40을 비교하고 동점은 grid 순서로 선택한다.
- 동일 유사도에서 자기 자신이 검색 결과에 없으면 이웃이 K+1개 남을 수 있던 경우를 수정했다. 0보다 큰 cosine 유사도를 계산한 뒤 자기 자신을 제외하고 최대 K개를 선택한다. 소수 12자리로 비교한 유사도가 같으면 place_id 사전순이다.
- score/predict의 후보 순서와 지원 불가(-inf/NaN) 처리를 검증했다. 순수 CF에 fallback은 추가하지 않았다.

## 실행 결과

Python 3.11.18, pandas 2.2.3, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.6.1에서 확인했다. Validation 선택은 K=40이었다. K를 정한 뒤 test는 같은 모델로 warm/all을 비교했다. 검증 중 다른 로컬 환경에서 얻은 진단값은 이 표에 넣지 않았다.

| 후보 모드 | 모델 | Recall@10 | NDCG@10 |
|---|---|---:|---:|
| warm | popularity | 0.0754 | 0.0396 |
| warm | item_knn_k40 | 0.0298 | 0.0156 |
| all | popularity | 0.0402 | 0.0211 |
| all | item_knn_k40 | 0.0159 | 0.0083 |

warm 행은 warm-positive 570명 기준, all 행은 positive 1,069명 기준이다. 전체 test 사용자는 1,243명이다. warm 후보는 7,172곳, all 후보는 고정 카탈로그의 16,764곳이며 개인 train 장소는 제외한다.

CF는 warm/all 모두 인기순보다 낮다. 후보를 넓혀도 순수 CF는 cold 장소에 점수를 못 내므로 추천 결과는 같고, 정답 분모가 달라 지표가 달라진다. Warm 실행의 전체 정답 기준 진단값은 Recall 0.0159 / NDCG 0.0083이며 all 실행의 전체값과 같다.

CF의 Top-10 제공 비율은 0.8129다. Positive 사용자 중 86명은 유효 점수가 하나도 없어 지표에 0점으로 포함했다. 평점 예측은 1,243쌍 중 143쌍만 직접 지원했다. Fallback 포함 MAE 0.6633 / RMSE 0.8223, 지원 쌍만의 MAE 0.6507 / RMSE 0.9907이다. 전체평균 fallback의 영향을 구분해서 해석한다.

기존 CF Recall 0.0131 / NDCG 0.0061을 재현했다고 보지는 않는다. 이웃 선택의 동점 규칙과 K 상한을 보완한 구현이므로 기존 수치와 달라질 수 있다. 이를 성능 개선의 증거로 해석하지 않는다. 원래 구현을 같은 로컬 환경의 구/신 평가기로 비교했을 때 랭킹 값은 같았으며, B의 원래 실행 환경·전체 점수 파일이 없어 원래 수치 차이의 원인을 확정하지는 못했다.

집계 수치·환경·데이터/후보/평가 분모 해시는 [cf_v1_1_summary.json](cf_v1_1_summary.json)에 있다. 모델 간 같은 모드의 데이터·후보·평가 대상 해시가 일치하는지도 확인했다. 개인별 점수는 로컬 outputs에만 저장했다.

## 재실행

```sh
python scripts/cf.py --year 2022 --tune --candidate-mode both
# K 확정 후 최종 test; 위 실행에서 K=40을 고른 경우
python scripts/cf.py --year 2022 --k 40 --split test --candidate-mode both
```

Hybrid에 전체 유효 점수를 넘길 때는 `--predictions-top 0`을 추가한다. Top-100 교집합으로 후보를 줄이지 않고 공통 후보에 맞춰 연결한다. 현재 통합 확인에서는 개인별 파일 크기를 줄이기 위해 top 10만 저장했고, 실제 Hybrid 점수 연결은 아직 하지 않았다.

원본과 전처리 스키마·분할은 변경하지 않았다. 전체 테스트 19개를 통과했다.
