# DateMate 공통 데이터 명세

생성 기준: `scripts/build_common_dataset.py` 실행 결과. 원본 AI-Hub CSV는 읽기 전용이며 수정하지 않는다.

## 확정 사항

- 원천 사용 범위: 2022·2023의 `TL_csv`(Training)와 `VL_csv`(Validation). 2022 `TS_csv`/`VS_csv`는 대응 파일과 SHA-256이 같은 복제본이므로 제외한다.
- 사용자 키: `user_id = year + ':' + TRAVELER_ID`. 서로 다른 연도의 동일 인물을 같은 사용자로 가정하지 않는다.
- 방문 키: `interaction_id = year + ':' + TRAVEL_ID + ':' + VISIT_AREA_ID`. `VISIT_AREA_ID`는 방문 기록 식별자이지 공유 장소 키가 아니다.
- 장소 키: POI가 있으면 `poi:<POI_ID>`. 없으면 이름+도로명주소, 이름+지번주소, 이름+좌표(소수 4자리)의 강한 정확 일치만 해시 키로 만든다. 비-POI 행이 기존 POI의 이름+주소/좌표에 유일하게 정확 일치하면 그 POI로 통합한다. 모호하거나 근거가 약한 행은 모델 상호작용에서 제외한다.
- 만족도 타깃: 방문지 테이블 `DGSTFN`의 1~5 정수. `REVISIT_INTENTION`과 `RCMDTN_INTENTION`은 보조 결과값이며 입력 특성으로 사용하지 않는다.
- DateMate 후보: 방문 유형 1~8, 10~13이면서 서울·경기·인천 주소 또는 행정코드를 가진 장소. 교통시설(9), 집·친지집·사무실·숙소(21~24)는 제외한다.

## 파일

### `outputs/processed/interactions.csv`

한 행은 한 방문 평가다. 모델 학습 전에 사용자–장소 쌍을 집계하려면 **먼저 분할한 뒤 각 분할 안에서만** 평균을 계산한다.

| 컬럼 | 의미 |
|---|---|
| `interaction_id` | 전역 방문 키 |
| `user_id` | 연도 접두어가 붙은 익명 사용자 키 |
| `place_id` | 연도 공통 장소 키 |
| `year`, `source_split` | 원천 연도와 AI-Hub 원본 분할(출처 추적용) |
| `travel_id`, `visit_area_id` | 원천 조인/감사용 키 |
| `visit_date`, `visit_order` | 사용자 내 시간순 분할 기준 |
| `rating` | 만족도 1~5 타깃 |
| `revisit_intention`, `recommendation_intention` | 보조 타깃; 입력 금지 |
| `visit_type_code` | 방문 유형 코드 |
| `companion_relation_codes`, `couple_relation` | 여행 단위로 먼저 요약한 동반자 맥락 |
| `known_user_split` | 사용자–장소 쌍을 묶은 시간순 train/validation/test 또는 평가 비대상 사용자의 `train_only` |
| `temporal_split` | 2022=train, 2023=test인 연도 이동 평가 |
| `is_warm_item_in_known_user_train` | 같은 연도 known-user train에 존재한 장소인지 여부 |

행 수: 2022 15,324, 2023 14,161, 합계 29,485.

### `outputs/processed/places.csv`

`place_id`당 한 행. 이름·주소·좌표·방문 유형·POI 분류와 품질 플래그를 제공한다. `is_model_candidate=1`만 추천 후보로 사용한다. 전체 34,671개 중 후보는 16,764개다. POI 분류는 연결된 행에서도 결측이 많으므로 이름 TF-IDF, 방문 유형, 지역, 좌표를 함께 사용한다.

### `outputs/processed/users.csv`

`user_id`당 한 행. 인구통계, 선호 지역 3개, `travel_style_1`~`8`을 보존한다. 여행 성향 축의 양 끝 의미는 공식 정의서 확인 전 임의로 이름을 붙이지 않는다.

### 감사 산출물

- `quality_report.json`: 결측·조인·희소도·연도 공통 장소·분할 통계.
- `source_inventory.csv`: 검사한 56개 TL/VL CSV의 파일명, 행/열 수, 헤더와 공통 데이터셋 사용 여부.

## 품질 판정

- 2022 방문→여행 연결 누락: 3,546행 (10.45%), 사용자 프로필 누락: 6,536행 (19.27%). 해당 행은 모델 상호작용에서 제외한다.
- 2023 방문→여행/사용자 연결 누락: 0/0행.
- 모델 상호작용 희소도: 2022 99.9444%, 2023 99.9448%. 단일 사용자 장소 비중도 높아 CF는 가능하지만 보조 모델로 취급한다.
- 공통 모델 장소: 1,609개. 연도 이동 평가는 전체와 공통 장소 하위집합을 따로 보고한다.
- 평점은 4~5점에 치우쳐 있으므로 전체평균·인기순·장소평균 기준선과 비교하고 MAE/RMSE만으로 결론내리지 않는다.

## 누수 없는 학습/평가

1. 기본 known-user 평가는 `known_user_split`을 사용한다. 동일 사용자–장소의 반복 방문은 반드시 한 분할에만 들어간다.
2. 마지막 장소를 test, 직전 장소를 validation으로 두고 나머지를 train으로 둔다. 고유 장소가 5개 미만인 사용자는 모든 행을 `train_only`에 두며 known-user 지표 분모에는 넣지 않는다.
3. CF 평가는 `is_warm_item_in_known_user_train=1` 하위집합을 주 지표로, CBF/hybrid는 전체와 warm-item 지표를 모두 보고한다.
4. 2022→2023 `temporal_split`은 모든 2023 사용자가 cold user다. 순수 CF 개인화 평가가 아니라 취향 프로필 기반 CBF/hybrid의 연도 이동 평가다.
5. TF-IDF 사전, 스케일러, 장소/사용자 평균, 인기순, 음성 샘플 후보는 train에서만 적합한다. 원본 `source_split`은 출처 정보일 뿐 최종 평가 분할로 사용하지 않는다.
6. 같은 사용자의 동반자 행을 방문지와 직접 다대다 조인하지 않는다. 이 전처리는 여행 단위 관계 코드 집합으로 요약한 뒤 방문에 연결한다.

## 모델 가능성

- CBF: 가능하고 1차 주력. 방문 유형 one-hot, 이름 TF-IDF, 지역/POI 카테고리, 좌표·거리, 사용자 여행 성향/선호 지역을 사용한다.
- CF: 구현은 가능하나 매우 희소하고 단일 사용자 장소가 많다. 인기순 및 item-mean 기준선보다 좋아지는지 검증될 때만 유지한다.
- Hybrid: CBF를 기본으로 하고 CF 신뢰도(이웃 수/지원도)가 충분한 warm user·warm item에만 CF 점수를 가중한다.
- 커플 결합: 원천 데이터는 두 조사 참여자의 프로필이 한 커플로 짝지어졌다는 근거가 없다. 실제 서비스에서는 두 사용자가 별도로 입력한 취향 점수의 평균/최솟값 등으로 결합하고, 원천 동반자 정보는 맥락 또는 하위집단 분석에만 쓴다.

## 협업 저장소 운영 보충 (2026-10-29)

기존 키·스키마·분할·기준 수치는 보존한다. 생성 스크립트의 기본 명세 출력만 `outputs/processed/data_spec.generated.md`로 바꿔 이 검토 문서의 자동 덮어쓰기를 방지한다.

`user_id`는 원본 ID에 연도를 붙인 가명 키이며 비식별성 보장이 아니다. 개별 기록은 업로드하지 않는다. 장소 전체기간 통계와 사용자 interaction_count는 모델 입력 금지다. 전체기간 장소 통합 카탈로그의 시간 정보 사용 한계와 정확한 후보·평가 분모는 `evaluation.md`에 정의한다. 기존 시간순 분할은 사용자–장소 마지막 방문 순서의 pair holdout이며 엄격한 전역 시간 분할은 아니다.
