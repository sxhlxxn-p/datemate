# Matrix Factorization 확장

Proposal의 선택 확장인 명시적 평점 MF를 구현했다. 기존 Item-KNN CF를 대체하지 않고 별도 모델로 비교한다. 기존 데이터·분할·warm/all 후보·지표를 유지한다.

## 모델

예측값은 `학습 전체 평균 + 사용자 편향 + 장소 편향 + 사용자 벡터·장소 벡터 내적`이다. 관측된 train 평점의 제곱 오차와 편향·벡터의 L2 정규화 합을 줄인다. 학습 전체 평균은 고정하고 사용자·장소 쪽을 번갈아 ridge least-squares로 갱신한다. 관측되지 않은 쌍을 0이나 부정 응답으로 넣지 않는다. 신경망·implicit-feedback ALS·결측값을 0으로 채운 SVD와 구분한다.

- 학습: 2022 `train + train_only`의 분할 내 평균 평점
- 후보 설정: 잠재 차원 8·16, 정규화 0.1·1·10, 20회 반복, seed 42
- 선택: warm validation의 `ndcg_at_10_warm`, 동점이면 작은 차원·작은 정규화 순서
- Test를 보고 차원·정규화·반복 수를 바꾸지 않는다. validation을 학습에 추가하지 않는다.
- 순위에는 원시 추정 평점을 사용한다. MAE/RMSE 계산 시 공통 평가기가 [1,5]로 제한한다.
- 학습에 없는 사용자·장소는 순위 점수 `-inf`, 평점 `NaN`으로 반환한다.
- 평점 예측이 불가능한 쌍은 공통 평가기가 train 평균으로 대체해 전체 지표를 계산한다. 직접 예측한 쌍만의 지표도 함께 보고한다.
- 편향만 사용하는 factors=0, reg=1 모델을 추가 진단으로 평가한다. 선택된 MF와 정규화가 다를 수 있어 잠재 벡터만의 효과를 분리하는 통제 실험으로 해석하지 않는다.

## 실행

기존 requirements 외에 추가 패키지는 필요 없다. 전처리 데이터를 준비한 뒤 순서대로 실행한다.

```sh
python scripts/run_mf.py --stage tune --protocol internal
python scripts/run_mf.py --stage evaluate --protocol internal
```

2022 내부 평가의 입력은 `outputs/processed/`다. 설정과 입력·소스 해시를 `outputs/matrix_factorization/internal/frozen_settings.json`에 고정한다. 이후 입력·설정·모델·평가 코드가 바뀌면 평가를 거부한다. `--input-dir`, `--output-dir`로 별도 위치를 지정할 수 있다.

`score(users, candidates)`와 `predict(users, places)` 인터페이스는 기존 모델과 같다. 추천 시 방문 제외는 공통 평가기가 담당한다. MF 모델에는 장소 메타데이터나 전체기간 통계를 입력하지 않는다.

## 신규 사용자 외부 평가

외부 평가 v2의 2022 전용 사전과 2023 support/query가 먼저 생성돼 있어야 한다. 이 데이터는 기존 내부 평가와 사전이 달라 **2022 validation에서 MF 설정을 별도로 선택**한다.

```sh
python scripts/run_mf.py --stage tune --protocol external
python scripts/run_mf.py --stage evaluate --protocol external
```

기본 입력은 `outputs/proposal_external_v2/data/`다. 분할 생성은 [외부 평가 안내](external_evaluation_v2.md)를 따른다. tune은 2022 validation만 사용한다. 2023은 평가할 때만 사용하며, 각 신규 사용자의 초기 평점 중 학습에 등장한 장소의 평점으로 사용자 편향·벡터만 ridge 추정한다(fold-in). 전체 평균·장소 편향·장소 벡터는 고정한다. 초기 입력에 학습 장소가 하나도 없으면 그 사용자는 MF로 점수를 낼 수 없다. Query 정답은 fold-in에 사용하지 않는다.

순수 MF와, 방문 장소를 제외한 뒤 MF 점수가 전혀 없는 사용자만 인기순으로 전환하는 정책을 분리해 보고한다. 점수가 있는 사용자에게 인기순을 섞거나 일부 개수를 채우지 않는다. 이 전환은 랭킹 정책이므로 별도의 평점 지표가 없다.

직접 예측 가능한 정답 쌍만의 비교에서는 모든 모델에 같은 정답 부분집합을 적용한다. 이 보조 결과를 전체 사용자 결과 대신 제시하지 않는다. 서로 다른 모델의 지원 쌍 수를 확인하지 않고 supported MAE를 직접 비교하지 않는다.

## 출력과 검증

원본·전처리·개별 예측은 로컬 `outputs/`에만 둔다. 상세 평가 경로는 `outputs/experiments/mf_extension/`, 집계는 `outputs/matrix_factorization/{internal,external}/summary.json`이다. 같은 후보 모드에서 모델 간 데이터·후보·정답 쌍·positive 사용자 해시가 일치하는지 검사한다. 기존 실험 출력은 덮어쓰지 않는다.

테스트는 목적함수 감소, seed 재현성, holdout 변경 시 학습 불변, 지원 불가 사용자·장소, fold-in의 ridge 해와 장소 벡터 고정, 방문 제외, 가상 데이터 내부·외부 전체 실행과 입력 변조 거부를 확인한다.

참고: Koren, Bell, Volinsky (2009), [Matrix Factorization Techniques for Recommender Systems](https://doi.org/10.1109/MC.2009.263).
