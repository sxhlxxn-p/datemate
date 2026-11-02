# 전처리 재현 확인

2026-10-29, POI master를 포함한 전체 실행 결과다.

| 항목 | 수 |
|---|---:|
| 상호작용 | 29,485행 |
| 사용자–장소 쌍 | 28,527쌍 |
| 상호작용이 있는 사용자 | 6,031명 |
| 전체 사용자 프로필 | 6,480행 |
| 장소 | 34,671개 |
| 추천 후보 | 16,764개 |
| 두 연도 공통 모델 장소 | 1,609개 |

| known_user_split | 행 수 |
|---|---:|
| train | 14,010 |
| train_only | 10,260 |
| validation | 2,600 |
| test | 2,615 |

원본 목록의 CSV 89개 경로·크기를 확인했고, 다시 생성한 `interactions.csv`, `places.csv`, `users.csv`, `source_inventory.csv`는 기존 파일과 SHA-256이 일치했다. 해시는 [reproduction_checks.json](../reports/reproduction_checks.json)에 있다.

기본키, 참조 관계, 평점 범위, 사용자–장소 분할 분리와 warm-item 플래그 검증을 통과했다. 가상 데이터 20행의 전체 실행, 반복 방문의 분할 유지, 잘못된 분할을 거부하는 테스트도 통과했다. 저장소를 새로 clone한 환경에서도 가상 데이터 실행을 확인했다.

위 수치는 공통 전처리 검증 결과이며 모델 성능 지표는 아니다.

2026-11-02에는 같은 데이터 해시와 clean Git 커밋 `ce0d136`에서 기준선·CBF·CF·Hybrid의 2022 test를 다시 실행하고, 2023 Hybrid test도 확정했다. 모델별 데이터·후보·평가 대상 해시가 같은지 자동 검사했으며 전체 테스트 37개를 통과했다. 결과는 [최종 모델 평가](../reports/Final_Model_Evaluation.md)와 [집계 JSON](../reports/final_evaluation_summary.json)에 분리해 기록했다. 전체기간 장소 통합에 따른 평가 범위는 [평가 기준](evaluation.md)을 참고한다.
