# 피드백 기반 추천 전략 선택

CBF·CF·Hybrid의 기존 정확도 평가와 별개의 확장이다. 한 번의 추천에서 네 가지 전략 중 하나를 고르고, 그 전략을 사용한 뒤 받은 피드백만 학습한다. 장기 경로를 최적화하는 전체 강화학습은 아니다.

## 구성

- 행동: `cbf_mean`, `cbf_min`, `hybrid_mean`, `hybrid_min`
- 맥락: 두 사람의 선호 차이, 초기 이력의 충분성, 장소 속성의 충분성을 각각 두 구간으로 나눈 8개 segment
- 정책: segment별 epsilon-greedy. 확률 0.1로 균등 탐색하며, 그 외에는 누적 보상의 Beta(1,1) 평활 평균이 큰 행동을 선택한다. 동점은 균등 선택한다.
- 피드백: 선택한 추천에 대한 0~1의 관측 보상. 모델 점수를 보상으로 사용하지 않는다.

`datemate.bandit.ContextualBandit`은 `select(context)`에서 행동과 실제 선택 확률을 반환하고, `update(context, action, reward)`에서 선택한 행동만 갱신한다. 사용자 ID는 맥락으로 쓰지 않는다. 현재 정책 상태는 메모리 내에 있으며 서버 저장·지연 피드백 처리·배포는 구현 범위 밖이다.

## 시뮬레이션 실행

추가 패키지나 AI-Hub 데이터가 필요하지 않다. 기존 requirements 환경에서 실행한다.

```sh
python scripts/run_bandit_simulation.py
```

설정은 `configs/bandit_simulation.json`, 출력은 `outputs/bandit_simulation/`이다.

- 3개 환경 × 20개 seed, 각 seed 학습 3,000회·분리된 평가 1,000회
- 매 회 독립적인 가상 사용자 쌍과 후보 40개, 추천 목록 5개
- 실제 CBF·CF를 재학습하지 않는다. 가상 잠재 선호에 잡음을 넣은 점수로 전략 선택을 실험한다.
- Hybrid alpha=0.4는 확장 실험의 고정 설정이며 기존 2022의 선택값을 바꾸지 않는다. 실제 결합 코드의 전체 재현 실험도 아니다.
- `informative`는 이력·속성이 충분할 때 추정 잡음이 작고, `noisy`는 잡음이 두 배다. `uninformative`는 점수가 잠재 선호와 독립이다.
- 각 장소에서 두 사람의 잠재 수락 확률을 곱하고 추천 목록에서 평균해, 목록에서 균등하게 한 장소를 노출했을 때의 공동 수락 확률을 정의한다. 그 확률의 Bernoulli 피드백을 생성한다. 이 수락 독립성·균등 노출 가정은 가상 환경의 가정이다.
- 동일 seed에서 모든 정책은 같은 맥락·잠재 선호·행동별 가상 피드백을 공유한다. 학습자에게는 선택 행동의 피드백만 전달한다.
- 평가에서는 갱신과 epsilon 탐색을 끄고 정책을 고정한다. 학습 결과를 본 뒤 설정을 다시 고르지 않았다.

`summary.json`에는 seed별·평균 보상, 기대 보상, 기대 regret과 paired seed bootstrap 구간을 저장한다. `synthetic_events.csv`에는 단계, 정책, 맥락, 행동, 선택 확률, 관측 보상을 기록한다. 원시 이벤트는 저장소에 올리지 않는다. Oracle은 시뮬레이터의 모든 잠재 보상을 아는 상한 진단용이며 실제 정책이 아니다. 일반 bandit은 선택하지 않은 행동의 보상을 알 수 없다.

## 실제 추천 코드와 연결

`rank_strategies(cbf_scores, hybrid_scores, candidate_ids, excluded, k)`는 기존 `couple_recommendations`를 사용해 네 전략의 추천 목록을 만든다. 두 점수 배열은 같은 후보 순서의 `(2, N)`이며 [0,1] 범위여야 한다. `excluded`에는 두 사용자의 초기 입력·방문 이력 합집합을 전달한다. 기존 모델의 원시 유사도나 평점을 그대로 전달하지 말고 기존 고정 범위 변환 계약에 맞춘다.

```python
from datemate.bandit import ACTIONS, ContextualBandit, rank_strategies

policy = ContextualBandit(seed=42)
rankings = rank_strategies(cbf_scores, hybrid_scores, candidate_ids,
                          excluded=both_users_seen, k=10)
action, propensity = policy.select(context_segment)
recommendations = rankings[ACTIONS[action]]
# 실제 응답을 받은 뒤에만 호출한다. 예: 두 사용자 모두 수락하면 1, 아니면 0.
policy.update(context_segment, action, observed_joint_feedback)
```

위 변수들은 실제 모델·입력·피드백에서 제공해야 한다. 실제 적용 전에는 충분한 추천 후보가 있는지, 두 사용자의 피드백이 모두 도착했는지 확인하고 이벤트 ID와 선택 확률을 함께 저장해야 한다. 응답 누락을 자동으로 거절로 간주하지 않는다. AI-Hub의 미방문 기록으로 거절을 만들지 않는다. 현재 실행한 결과는 실제 서비스나 실제 사용자 실험이 아니다.

## 해석

시뮬레이션 결과는 가정한 보상·잡음·사용자 분포에서 정책이 동작하는지 확인한 결과다. 기존 Recall/NDCG나 실제 커플 만족도 개선으로 해석하지 않는다. 음수나 차이가 없는 비교도 모두 남긴다. 실제 검증에는 추천 노출·선택 확률·두 사용자 피드백을 수집한 별도 실험이 필요하다.

참고: [Contextual Bandit 문제 정의](https://arxiv.org/abs/1402.0555), [선택 확률을 기록한 정책 평가](https://www.microsoft.com/en-us/research/publication/sample-efficient-nonstationary-policy-evaluation-for-contextual-bandits-2/).
