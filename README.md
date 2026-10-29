# DateMate — ML Team Project

서울·수도권 데이트 장소 추천 프로젝트 리포지토리. 공통 데이터 생성은 구현되어 있으며 **추천 모델 학습과 성능 검증은 아직 진행 전**입니다. -->역할 정해서 진행하면 될 거 같습니다.
실험 설정=/ 모델 구현체

## 먼저 실행하기 — 실제 데이터 없이

Python 3.10 이상을 사용합니다. 공통 전처리와 검증은 표준 라이브러리만 사용합니다.

```sh
git clone https://github.com/sxhlxxn-p/datemate.git
cd datemate
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/make_synthetic_data.py
python scripts/build_common_dataset.py --data-dir outputs/synthetic_source --output-dir outputs/synthetic_processed --spec-path outputs/synthetic_processed/data_spec.generated.md --skip-poi-master
python scripts/validate_common_dataset.py --input-dir outputs/synthetic_processed
```

가상 샘플은 코드로 생성하며 AI-Hub 기록을 복사하거나 변형해 만들지는 않아서 샘플 성공이 실제 데이터의 성능이나 품질을 의미하지 않습니다.

## 실제 데이터 준비 및 재현

1. 각 팀원이 AI-Hub에서 국내 여행로그 데이터(수도권) 2022·2023 자료의 이용 신청·승인을 받아 직접 다운로드합니다.
   [데이터 준비 및 이용 조건](docs/data_access.md)해서 확인 가능합니다. 
3. [data/README.md](data/README.md)의 폴더 구조로 압축 해제합니다. 파일명·크기는 [manifest](docs/data_manifest.json)에 있습니다.
4. 다음 명령으로 검사하고 생성합니다.

```sh
python scripts/check_data.py
python scripts/build_common_dataset.py
python scripts/validate_common_dataset.py
```

외부 데이터 위치도 지원합니다. 기존 원본 폴더를 사용할 경우 check_data.py와 build_common_dataset.py에 각각 `--data-dir "/absolute/path/to/data"`를 지정하세요. 기본 생성 위치는 `outputs/processed/`입니다. POI master 전체 스캔이 포함되므로 시간이 걸립니다. `--skip-poi-master`는 빠른 진단용이며 공식 전체 실행 결과와 비교하지 않습니다.

생성물은 `interactions.csv`, `places.csv`, `users.csv`, `quality_report.json`, `source_inventory.csv`, `data_spec.generated.md`입니다. 검토된 [공통 명세](docs/data_spec.md)는 자동 덮어쓰지 않습니다. 원본 data 아래로 출력하는 경로는 거부합니다.

기존 검증 보고서의 기준은 상호작용 **29,485행**, 장소 **34,671개**, 후보 **16,764개**입니다. 실제 재현 확인 결과는 [검증 기록](docs/verification.md)에 기록합니다. 과거 EDA 수치와 정제 후 공통 데이터 수치를 구분하세요.

## 모델 실험 시작

[평가 명세](docs/evaluation.md)와 [공통 설정](configs/experiment.json)을 모든 모델에서 공유합니다. 첫 비교는 2022 known-user train/validation/test로 진행합니다. 2023 within-year 평가는 별도 표에, 2022→2023 cold-user 평가는 별도 실험으로 보고합니다.

- CBF 담당: 이름 TF-IDF, 방문 유형, 지역·좌표 기반 점수. 사전과 스케일러는 train에서만 학습.
- CF 담당: item-based KNN, 이웃 수·지원도와 추천 가능 비율 보고.
- Hybrid 담당: validation에서만 가중치 조정, CF 불가 시 CBF fallback. CBF/CF 점수 척도 보정도 train/validation만 사용.
- 공통 담당: 인기순·평균 기준선, 동일 후보 집합 및 평가 분모 확정, 코드·설정·환경·집계 지표 통합.

실제 커플 공동 만족도 정답은 없습니다. 개인 추천 평가와 가상 사용자 쌍 시연을 분리합니다. 상세 작업·최종 산출물은 [협업 안내](CONTRIBUTING.md)참고해주세요.

## 저장소 내용과 데이터 보호

코드·문서·설정·집계 통계만 공유합니다. 원본 8.5GB, ZIP, 실제 방문/프로필/장소 CSV, 모델 파일, 노트북 출력, 문서 원본, API 키는 업로드하지 않습니다. 원본 식별자에 연도를 붙인 user_id는 가명 키입니다.

- [공통 스키마와 분할](docs/data_spec.md)
- [초기 EDA 및 공통 구조 보고서](reports/dataset_validation.md)
- [평가와 누수 방지 규칙](docs/evaluation.md)
- [데이터 출처·권한](docs/data_access.md)
- [로컬·원격 검증 기록](docs/verification.md)

기존 `scripts/eda.py`는 감사용 초기 EDA입니다. 그 결과를 모델 입력으로 사용하지 마세요. 원본 출처 Training/Validation은 최종 평가 분할이 아닙니다. 배포 코드에 별도 오픈소스 라이선스를 부여하지 않았으며 AI-Hub 데이터의 이용 조건은 별도로 적용됩니다.
