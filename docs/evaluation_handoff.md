# 평가 코드 적용 순서

공통 기준은 [evaluation.md](evaluation.md) v1.1이다. 기존 데이터와 분할은 바꾸지 않는다. 이전 결과는 보관하고 v1.1로 다시 측정한 결과끼리 비교한다.

## 공통

```sh
python scripts/protocol_summary.py --year 2022
python scripts/eval_protocol.py --year 2022 --split validation --candidate-mode warm
python scripts/eval_protocol.py --year 2022 --split validation --candidate-mode all
```

2022 test는 전체 평가 사용자 1,243명, positive 사용자 1,069명, warm-positive 사용자 570명이다. 모델이 추천할 수 있는 사용자만 뽑아서 지표를 계산하지 않는다. 추천하지 못하면 해당 분모에 남겨 두고 0점 처리한다.

## A — CBF

```sh
python scripts/cbf.py --year 2022 --candidate-mode both
```

warm validation NDCG@10으로 설정을 고르고, 같은 설정으로 warm/all 결과를 저장한다. 설정 확정 후 `--final-test`를 추가한다. 후보 확장 시 TF-IDF·one-hot을 다시 fit하지 않는다.

## B — CF

ItemKNN 모델과 실행부에 v1.1을 연결했다. `python scripts/cf.py --year 2022 --tune --candidate-mode both`로 warm validation에서 k를 선택하고 같은 모델의 warm/all 결과를 저장한다. 설정 확정 후 `--final-test`를 추가한다. 기존 k를 재현하려면 `--k 40 --split test --candidate-mode both`를 사용한다.

CF가 점수를 내지 못한 조합은 `-inf`로 둔다. 인기순 등 fallback을 넣은 모델은 별도 이름으로 비교한다. warm에서 고른 k를 all에도 그대로 쓴다.

## C — Hybrid

A/B의 동일한 연도·split·candidate_mode 점수를 받는다. 직접 score()를 호출하거나 `--predictions-top 0`으로 전체 유효 점수를 저장한다. Top-100의 교집합만 쓰면 평가 후보가 바뀐다.

파일을 연결할 때는 전체 후보를 유지한 채 user_id/place_id로 정렬한다. 후보는 공통 데이터에서 복원하고 CF 출력에서 빠진 조합은 추천 불가로 처리한다. inner join으로 사용자·후보를 삭제하지 않는다. train 제외 장소는 계속 제외한다. 스케일 보정과 가중치 선택은 train/validation에만 맞춘다.

실제 모델 객체의 전체 후보 점수를 연결하고 단위·실데이터 재현 검증을 완료했다. 실행 방법과 2022·2023 결과는 [Hybrid 실험 보고서](../reports/Hybrid_Experiment_Results.md)에 정리했다. 가상 예제 CSV는 동작 확인에만 사용하고 성능 결과로 사용하지 않는다.

## 결과 공유

`outputs/experiments/v1_1/known_user_2022/<warm|all>/<model>/<split>/`의 metrics.json과 config.json에서 개인 기록 없는 집계만 공유한다. 모델명, 파라미터, 분할, 후보 모드, 평가 분모, 데이터/후보/평가 쌍 해시를 함께 확인한다. 개인별 예측 CSV는 GitHub에 올리지 않는다.
