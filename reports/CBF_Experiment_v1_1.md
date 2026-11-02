# CBF 공통 평가 재실험

2026-11-01, main의 CBF 코드를 수정하지 않고 공통 평가 v1.1로 2022 실험을 다시 실행했다. 기본 CBF와 지역 특성 추가 모델을 각각 validation에서 튜닝하고, 선택한 설정을 고정해 warm/all test를 평가했다.

## 환경과 조건

- Python 3.11.18 / pandas 2.2.3 / numpy 2.2.6 / scipy 1.15.3 / scikit-learn 1.6.1
- train + train_only로 학습하고, 같은 분할 안에서 반복 평점을 평균 처리
- TF-IDF·one-hot은 train 장소로만 fit. all 후보는 transform만 수행
- warm validation NDCG@10으로 선택. 동점은 기존 grid의 첫 설정
- Top-10, 평점 평균 4점 이상을 정답으로 사용. 사용자 train 장소는 후보에서 제외
- test 전체 1,243명 중 positive 1,069명, warm-positive 570명
- 후보는 warm 7,172곳 / all 16,764곳(개인별 train 장소 제외 전)

기본 CBF 16개, 지역 추가 CBF 48개로 총 64개 설정을 비교했다. 두 모델의 설정과 모델 간 선택 모두 validation만 사용했다.

## 선택된 설정

| 모델 | 이름 TF-IDF | 방문 유형 가중치 | 지역 가중치 | 사용자 이력 가중치 | Warm validation NDCG@10 |
|---|---|---:|---:|---|---:|
| 기본 CBF | char, 2~3글자 | 0 | 0 | 원래 평점 | 0.0179 |
| 지역 추가 CBF | char, 2~3글자 | 0.5 | 0.5 | 원래 평점 | 0.0266 |

이름·방문 유형을 조합하는 기본 모델의 grid에는 방문 유형 가중치 0도 포함돼 있다. 이번에는 이름만 쓰는 설정이 선택됐다.

두 모델 중 **지역 추가 CBF(cbf_region)를 선택**한다. 이는 test 수치가 아니라 warm validation NDCG 기준이다. Hybrid에 연결할 설정은 `name_mode=char, w_type=0.5, w_region=0.5, weighting=raw`이다.

## Test 비교

| 모델 | Warm Recall@10 | Warm NDCG@10 | All Recall@10 | All NDCG@10 |
|---|---:|---:|---:|---:|
| Popularity | 0.0754 | 0.0396 | 0.0402 | 0.0211 |
| CF, K=40 | 0.0298 | 0.0156 | 0.0159 | 0.0083 |
| 기본 CBF | 0.0175 | 0.0101 | 0.0168 | 0.0073 |
| 지역 추가 CBF | 0.0421 | 0.0193 | 0.0206 | 0.0116 |

Warm 열은 warm 후보와 warm-positive 570명, All 열은 전체 후보와 positive 1,069명 기준이다. 서로 분모·후보가 다르므로 모드 안에서 비교한다. 전체평균·장소평균 기준선의 랭킹 지표는 두 모드 모두 0이었다.

지역 추가 CBF는 이번 test에서 CF보다 높았고 인기순보다는 낮았다. 이후 Hybrid의 alpha를 validation에서 선택한 결과 2022에서는 0.00이 선택돼 Hybrid 순위와 지표가 지역 추가 CBF와 정확히 같았다. Hybrid 고유의 개선으로 해석하지 않는다. 현재 all 실험은 두 연도 로그로 만든 고정 카탈로그를 사용하므로 실제 시점의 신규 장소 성능으로 해석하지 않는다.

CBF 두 모델 모두 유효 추천 10개 제공 비율은 1.0이었다. 기본 CBF의 item coverage는 warm 0.4689 / all 0.2727, 지역 추가 모델은 warm 0.4905 / all 0.1305였다. 이 값은 점수가 유한한지에 따른 제공 비율이며, 높은 추천 정확도나 모든 사용자의 충분한 취향 근거를 뜻하지 않는다. 코사인 점수 0인 후보도 동점 규칙으로 순위에 들어갈 수 있다.

## 평점 예측

| 모델 | 전체 MAE | 전체 RMSE | 직접 지원한 쌍 / 1,243 |
|---|---:|---:|---:|
| 전체평균 | 0.6643 | 0.8026 | 1,243 |
| 기본 CBF | 0.6422 | 0.8234 | 291 |
| 지역 추가 CBF | 0.6164 | 0.8642 | 1,066 |

전체 지표는 미지원 쌍에 train 전체평균을 채운 값이다. 지역 추가 CBF는 전체평균보다 MAE가 낮지만 RMSE는 높아, 평점 예측 전반이 개선됐다고 단정하지 않는다. 지원 쌍만의 지표는 집계 JSON에 함께 기록했다.

## 확인과 재실행

CBF·지역 추가 CBF·CF·세 기준선의 데이터, 후보, 평가 쌍과 사용자 분모 해시가 같은 모드에서 일치했다. 실행 환경 버전도 일치한다. 모델·평가 코드·분할은 변경하지 않았다.

```sh
python scripts/cbf.py --year 2022 --candidate-mode both --final-test --predictions-top 10
```

선택 설정, 64개 validation 결과, 각 실행의 환경·해시·지표는 [cbf_v1_1_summary.json](cbf_v1_1_summary.json)에 있다. 최초 재실행은 기존 비추적 파일 때문에 git dirty가 기록됐지만, D파트 최종 확인에서는 clean 커밋 `ce0d136`에서 CBF와 다른 모델을 모두 다시 실행했다. 최종 비교는 [최종 평가 보고서](Final_Model_Evaluation.md)와 [검증된 집계 JSON](final_evaluation_summary.json)을 기준으로 한다.

개인별 예측은 로컬 outputs에만 저장했다. 이번 top-10 출력은 결과 확인용이다. C파트는 전체 점수 파일을 별도 생성하거나 `CBF(...).fit()` 후 공통 후보에 `score()`를 직접 호출해 연결한다. Top-10/100 결과 교집합으로 평가 후보를 줄이지 않는다.
