# 로컬 데이터 준비

원본 CSV는 GitHub에 포함하지 않습니다. 현재 배치는 다음과 같습니다.

```text
data/
  국내여행로그데이터(2022)/
    TL_csv/
    VL_csv/
    TS_csv/
    VS_csv/
  국내여행로그데이터(2023)/
    TL_csv/
    VL_csv/
    메타데이터/
```

정확한 CSV 상대 경로와 바이트 크기는 `../docs/data_manifest.json`을 참고하세요. `python scripts/check_data.py`로 준비 상태를 확인합니다. 한글 파일명의 NFC/NFD 차이는 비교 시 정규화합니다. 바이트 크기는 확인하지만 파일 내용의 해시까지 검증하지는 않습니다.

AI-Hub https://aihub.or.kr/ 에서 두 연도 수도권 여행로그 CSV를 확보하여 위와 같이 압축 해제하세요. 2022년 원천/라벨 자료는 중복되므로 EDA에는 TL_csv와 VL_csv만 사용합니다. POI master 전체 정밀 검사는 기존 보고서의 별도 스냅샷이며 기본 EDA 실행에서는 대용량 master를 읽지 않습니다.

GitHub 저장소를 clone하는 것만으로 데이터까지 내려오지는 않습니다. 각 팀원이 직접 승인받아 다운로드하고 인증키를 저장소에 올리지 마세요.

## 2026-10-29 협업 정책 보충

공용 클라우드로 원본/가공 CSV를 전달하지 않습니다. 각 팀원이 직접 이용 신청·승인 후 다운로드합니다. `docs/data_access.md`를 참고하세요. GitHub의 private 설정도 재배포 허가가 아닙니다. 전처리 시 POI master 전체 스캔은 기본 활성화됩니다. `--skip-poi-master` 결과는 진단용입니다.
