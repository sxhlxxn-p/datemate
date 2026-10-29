# 데이터 준비

AI-Hub의 국내 여행로그 데이터(수도권) 2022·2023을 각자 이용 승인 후 다운로드한다. 데이터 이용 조건은 [이용 안내](../docs/data_access.md)를 참고한다.

```text
data/
  국내여행로그데이터(2022)/
    TL_csv/
    VL_csv/
  국내여행로그데이터(2023)/
    TL_csv/
    VL_csv/
    메타데이터/
      tn_poi_master_POIMaster_all.csv
```

전처리는 두 연도의 TL/VL을 사용한다. 2022 TS/VS는 중복 자료라 사용하지 않는다. POI master는 위 파일을 우선 읽고, 없으면 TL_csv의 `tn_poi_master*.csv`를 찾는다.

```sh
python scripts/check_data.py
python scripts/build_common_dataset.py
python scripts/validate_common_dataset.py
```

`check_data.py`는 [최초 파일 목록](../docs/data_manifest.json)의 89개 파일 경로와 크기를 비교한다. 이 목록에는 2022 TS/VS도 들어 있어 해당 중복 파일을 생략한 경우 missing이 표시된다. 전처리 필수 파일 누락과 구분해서 확인한다. 한글 파일명은 NFC/NFD를 정규화해 비교하며, 파일 내용 해시까지 검사하지는 않는다.

데이터가 다른 폴더에 있으면 다음처럼 지정한다.

```sh
python scripts/check_data.py --data-dir "/path/to/data"
python scripts/build_common_dataset.py --data-dir "/path/to/data"
```

원본은 수정하지 않고 결과를 `outputs/processed/`에 저장한다. 기본 실행은 POI master 전체를 읽는다. `--skip-poi-master`는 빠른 동작 확인에만 사용한다.

원본·가공 데이터는 저장소나 공용 클라우드에 올리지 않는다. 다운로드 키도 각자 관리한다.
