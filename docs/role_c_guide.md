# C 담당 구현 및 인수인계

## 현재 구현 범위

Hybrid 점수 결합, CF 미지원 시 CBF 대체, 두 사용자 점수의 평균/최솟값 결합, CSV 기반 터미널 시연을 구현했다. 예제 데이터는 전부 가상이다. 실제 AI-Hub 학습, 성능 측정, 최적 alpha 선택, 웹 화면은 아직 수행하지 않았다.

## 바로 실행

프로젝트 루트에서 실행한다. 추가 패키지 설치는 필요 없다.

```sh
python -m unittest discover -s tests -v
python scripts/run_hybrid.py --cbf examples/cbf_scores.csv --cf examples/cf_scores.csv --alpha 0.5 --output outputs/experiments/synthetic/hybrid/predictions.csv --couple 2022:demo_a 2022:demo_b --synthetic-demo --k 3
```

하나의 Hybrid 점수 CSV와 평균/최솟값 각각의 커플 순위 CSV를 생성한다. 터미널에는 두 사람의 점수도 표시한다. 예제의 alpha=0.5는 임의 시연 설정이며 validation으로 선택한 값이 아니다.

## A B D와 먼저 합의할 계약

A와 B는 각각 `user_id,place_id,score,split,year` 열이 있는 CSV를 전달한다. D가 정한 동일 연도/분할과 공통 후보를 사용한다. 상위 K개만 전달하면 안 된다. 사용자 자신의 train 방문 장소는 후보에서 제외한다. 같은 키가 중복되면 오류 처리한다. 프로토콜마다 파일을 분리하고 서로 섞지 않는다.

- A: 공통 후보 전체에 대한 CBF 점수. 기본 설정은 코사인 유사도 [-1,1]이다. 실제 모델이 다른 척도면 반드시 알려주고 bounds를 바꾼다. CBF가 점수를 못 내는 사용자의 처리 방식은 A/D와 합의해야 하며 C 코드는 누락을 임의 보충하지 않는다.
- B: CF가 실제로 지원하는 점수만 전달한다. 기본 설정은 평점 예측 [1,5]다. 미지원은 행 생략 또는 score 공란으로 표현한다. 이미 인기순/평균 fallback으로 채운 값을 순수 CF 점수처럼 보내면 안 된다. 최소 이웃/지원도 판정은 B가 담당한다.
- D: 모든 모델의 후보와 분모 일치 여부를 확인한다. CBF가 후보 전체를 제공하는지 확인하고 공통 평가 함수로 validation 지표를 계산한다. 평가 코드와 baseline을 C에서 중복 구현하지 않는다.

범위를 벗어난 점수와 NaN은 오류다. 필요하면 CLI의 `--cbf-bounds 0 1`처럼 명시한다. CF에 없는 후보는 CBF만 쓰며, CF에만 있는 후보는 계약 불일치로 오류 처리한다.

## 계산 이해

기본 고정 변환은 CBF `(score+1)/2`, CF `(score-1)/4`이다. 그 뒤 `alpha*CF+(1-alpha)*CBF`를 계산한다. 고정 범위 변환이므로 test 정답이나 전체 데이터 통계로 스케일러를 적합하지 않는다. 다만 두 모델 점수의 통계적 의미까지 동일하게 보정하는 것은 아니며 실제 validation 결과로 적절성을 확인해야 한다.

커플 평균은 두 점수의 평균이고, 최솟값은 낮은 쪽 점수다. 같은 장소에 두 사용자 점수가 모두 존재해야 한다. 동점은 place_id 사전순이다. CBF 근거를 CF/Hybrid 효과로 해석하지 않으며 현재 출력의 score_source는 계산 경로를 설명한다. 장소명/유형에 따른 자연어 추천 근거는 A의 속성 근거를 연결한 뒤 추가한다.

Hybrid 점수는 [0,1] 순위 점수이며 1~5 평점 예측이 아니다. 이 점수에 MAE/RMSE를 적용하지 않는다. CF 평점 오차는 B/D가 따로 계산한다.

## 실제 실험 순서

1. 주 실험은 2022 known_user_split만 사용한다. train/train_only로 학습하고 분할 내에서 반복 평점을 집계한다.
2. A/B의 validation 전체 후보 점수를 각각 준비한다.
3. alpha 0, 0.25, 0.5, 0.75별로 run_hybrid.py를 실행한다. 각 출력 경로를 다르게 지정한다.
4. D의 공통 평가 코드로 각 출력의 NDCG@10/Recall@10 및 전체/warm 지표와 coverage를 계산한다. 선택 주 지표와 동점 규칙은 결과를 보기 전에 팀에서 정한다. 예: validation 전체 NDCG@10 최대, 동점이면 작은 alpha. 이는 제안이며 기존 확정 규칙이 아니다.
5. 선택한 alpha를 고정하고 test 점수를 한 번 평가한다. test를 보면서 다시 조정하지 않는다.
6. 2023 연도 내 평가와 2022→2023 평가는 모델/출력 경로를 분리한다. 후자는 2023 평가 평점을 프로필로 사용하지 않는다. 자동 이력 제외 CLI는 known-user validation/test만 지원하므로 연도 이동은 D/A와 입력 프로필·후보 계약부터 맞춘다.

실제 커플 시연 예시:

```sh
python scripts/run_hybrid.py --cbf outputs/experiments/known2022/cbf/predictions.csv --cf outputs/experiments/known2022/cf/predictions.csv --alpha 0.25 --output outputs/experiments/known2022/hybrid/predictions.csv --couple 2022:USER_A 2022:USER_B --interactions outputs/processed/interactions.csv
```

위 사용자 ID와 alpha는 실제 검증 결과로 교체한다. 이 CLI는 전달된 두 사용자 점수가 한 연도/분할인지 확인하고 양쪽 train 방문 장소 합집합을 제외한다. 실험 평가용 개인 점수 파일은 모든 사용자를 유지한다. 실제 공동 만족도 정답은 없으므로 커플 시연을 실제 만족도 향상 증거로 주장하지 않는다.

## 남은 작업과 분담

C: A/B 점수 척도 확정 후 연결, validation 가중치 실험, 평균/최솟값 차이 시연, 결과 해석 및 발표 설명.
A/B: 실제 모델과 점수 파일 제공, 예측 가능 조건/점수 의미 설명.
D: 공통 평가 실행/연결, 후보와 분모 검증, 결과표 통합과 전체 코드 검수.

개인별 점수·실제 데이터·모델은 Git에 올리지 않는다. 예제 CSV만 허구 데이터로 공유 가능하다. 구현은 이번 작업에서 작성했으며 외부 코드 복사는 없다. 사용 표준 라이브러리는 argparse(CLI 인수), csv(CSV 입출력), pathlib(경로), math(유한수 검사), unittest(동작 검증), sys(루트 모듈 경로 설정)다.
