"""Generate fictional source-shaped CSVs locally; contains no AI-Hub records."""
import argparse
import csv
from pathlib import Path


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def generate(root):
    for year in ("2022", "2023"):
        for split in ("TL_csv", "VL_csv"):
            base = root / f"synthetic_{year}" / split
            users = [{"TRAVELER_ID": f"SYNTH_U{u}", "GENDER": "", "AGE_GRP": ""} for u in (1, 2)]
            trips = [{"TRAVEL_ID": f"SYNTH_T{u}", "TRAVELER_ID": f"SYNTH_U{u}"} for u in (1, 2)]
            companions = [{"TRAVEL_ID": f"SYNTH_T{u}", "REL_CD": "8"} for u in (1, 2)]
            visits = []
            for u in (1, 2):
                # One user's four places exercise train_only; the other's six exercise holdout.
                for i in (range(1, 7) if u == 1 else range(1, 5)):
                    if (i % 2 == 1) != (split == "TL_csv"):
                        continue
                    visits.append({"TRAVEL_ID": f"SYNTH_T{u}", "VISIT_AREA_ID": f"SYNTH_V{u}_{i}",
                        "POI_ID": f"SYNTH_P{i}", "VISIT_AREA_NM": f"가상장소{i}",
                        "ROAD_NM_ADDR": f"서울 가상주소 {i}", "LOTNO_ADDR": "",
                        "X_COORD": "127.0", "Y_COORD": "37.5", "SGG_CD": "11000",
                        "VISIT_AREA_TYPE_CD": "1", "DGSTFN": str(3 + i % 3),
                        "VISIT_START_YMD": f"{year}-01-{i:02d}", "VISIT_ORDER": str(i)})
            for name, rows in (("tn_traveller_master", users), ("tn_travel_info", trips),
                               ("tn_companion_info", companions), ("tn_visit_area_info", visits)):
                write(base / (name + "_synthetic.csv"), rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic_source"))
    args = parser.parse_args()
    generate(args.output_dir)
    print("Fictional inputs written to", args.output_dir)
