# DateMate

서울·수도권 데이트 장소 추천 프로젝트. AI-Hub 여행로그 2022·2023 데이터를 사용한다.
현재 공통 전처리까지 완료했다. 역할을 정한 뒤 이 데이터를 기준으로 CBF·CF·Hybrid를 구현하고 비교하면 된다.

## 실행 환경

Python 3.10 이상.

```sh
git clone https://github.com/sxhlxxn-p/datemate.git
cd datemate
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 데이터 준비와 전처리

각자 AI-Hub에서 이용 승인 후 데이터를 내려받아 `data/`에 넣는다. 폴더 구성은 [데이터 준비 안내](data/README.md)를 참고한다. 원본과 전처리 결과는 GitHub에 올리지 않는다.

```sh
python scripts/check_data.py
python scripts/build_common_dataset.py
python scripts/validate_common_dataset.py
```

다른 위치에 데이터가 있으면 앞의 두 명령에 `--data-dir "/path/to/data"`를 붙인다. `check_data.py`는 최초 확보한 전체 파일 목록을 검사하므로 중복 TS/VS 파일까지 확인한다. 전처리에는 TL/VL과 POI master만 사용한다.

결과는 `outputs/processed/`에 생성된다.

| 파일 | 내용 |
|---|---|
| `interactions.csv` | 사용자–장소 평점과 평가 분할 |
| `places.csv` | 장소 정보와 추천 후보 여부 |
| `users.csv` | 사용자 프로필 |
| `quality_report.json` | 조인 누락, 제외 사유, 분할 통계 |
| `source_inventory.csv` | 입력 CSV 목록과 스키마 |
| `data_spec.generated.md` | 해당 실행의 데이터 명세 |

전체 실행 기준 상호작용 29,485행, 장소 34,671개, 추천 후보 16,764개다. POI master를 읽는 데 시간이 걸린다. `--skip-poi-master`는 빠른 확인용이며 전체 실행과 결과가 다를 수 있다.

## 실험 진행

모델 입력은 위 공통 데이터로 통일한다. `scripts/eda.py`의 초기 분석 결과는 모델 입력으로 쓰지 않는다.

1. 2022의 `known_user_split`으로 먼저 비교한다. `train`·`train_only`로 학습하고 `validation`으로 튜닝한 뒤 `test`로 평가한다.
2. 전체평균·장소평균·인기순을 기준선으로 두고 CBF, item-KNN CF, Hybrid를 비교한다.
3. MAE/RMSE, NDCG@10/Recall@10, 추천 가능 비율과 coverage를 함께 기록한다.
4. 2023 연도 내 실험과 2022→2023 평가는 별도로 정리한다.

반복 방문 집계는 분할 후에 한다. TF-IDF·스케일러·평균·인기도는 train에서만 계산한다. `places.csv`의 전체기간 평균 평점·방문 수와 `users.csv`의 `interaction_count`는 입력 특성에서 제외한다.

후보 집합과 지표 계산은 [평가 기준](docs/evaluation.md), 파라미터 범위는 [실험 설정](configs/experiment.json)을 따른다. 역할별 작업과 PR에 넣을 내용은 [협업 안내](CONTRIBUTING.md)에 정리했다.

## 데이터 없이 동작 확인

아래 샘플은 실제 기록을 사용하지 않은 가상 데이터다. 공통 전처리는 별도 패키지 없이 실행된다.

```sh
python -m unittest discover -s tests -v
python scripts/make_synthetic_data.py
python scripts/build_common_dataset.py --data-dir outputs/synthetic_source --output-dir outputs/synthetic_processed --spec-path outputs/synthetic_processed/data_spec.generated.md --skip-poi-master
python scripts/validate_common_dataset.py --input-dir outputs/synthetic_processed
```

## 참고

- [공통 데이터 명세](docs/data_spec.md)
- [데이터 검증 보고서](reports/dataset_validation.md)
- [재현 확인 결과](docs/verification.md)
- [데이터 이용 안내](docs/data_access.md)

실제 커플의 공동 만족도 정답은 없어 개인 추천 평가와 사용자 쌍 시연은 구분한다. 장소 정보는 전체기간 로그로 통합했으므로 엄격한 시점별 평가에는 추가 전처리가 필요하다. 자세한 범위는 평가 기준에 적었다.
