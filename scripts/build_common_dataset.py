#!/usr/bin/env python3
"""Build a leakage-aware common DateMate dataset from AI-Hub travel logs.

The source data is read-only.  This script reads only TL_csv/VL_csv (the 2022
TS_csv/VS_csv trees are byte-identical duplicates) and writes derived files to
outputs/processed plus a generated data specification.

No third-party package is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SPLITS = {"TL_csv": "train", "VL_csv": "validation"}
YEARS = ("2022", "2023")
RECOMMENDABLE_TYPES = {str(i) for i in range(1, 9)} | {"10", "11", "12", "13"}
EXCLUDED_TYPES = {"9", "21", "22", "23", "24"}
CAPITAL_PREFIXES = ("서울", "경기", "인천")
CAPITAL_SIDO_CODES = {"11", "28", "41"}
PLACE_FIELDS = (
    "VISIT_AREA_NM",
    "POI_NM",
    "ROAD_NM_ADDR",
    "LOTNO_ADDR",
    "X_COORD",
    "Y_COORD",
    "SGG_CD",
    "VISIT_AREA_TYPE_CD",
)


def nfc(value: object) -> str:
    return unicodedata.normalize("NFC", "" if value is None else str(value))


def clean(value: object) -> str:
    return nfc(value).strip()


def norm_text(value: object) -> str:
    value = unicodedata.normalize("NFKC", clean(value)).casefold()
    return "".join(ch for ch in value if ch.isalnum())


def stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(nfc(p) for p in parts)
    return f"{prefix}:{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:20]}"


def to_int(value: object) -> int | None:
    try:
        number = int(float(clean(value)))
    except (TypeError, ValueError):
        return None
    return number


def to_float(value: object) -> float | None:
    try:
        number = float(clean(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def coord_key(x: object, y: object) -> str:
    xx, yy = to_float(x), to_float(y)
    if xx is None or yy is None:
        return ""
    return f"{xx:.4f},{yy:.4f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{k: clean(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def year_dir(data_dir: Path, year: str) -> Path:
    matches = [p for p in data_dir.iterdir() if p.is_dir() and year in nfc(p.name)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one directory for {year}, found {matches}")
    return matches[0]


def table_path(root: Path, split_dir: str, prefix: str) -> Path:
    matches = [p for p in (root / split_dir).glob("*.csv") if p.name.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {prefix} file in {root / split_dir}, found {matches}")
    return matches[0]


def mode(values: Iterable[str]) -> str:
    values = [clean(v) for v in values if clean(v)]
    if not values:
        return ""
    counts = Counter(values)
    return sorted(counts, key=lambda x: (-counts[x], x))[0]


def is_capital_area(row: dict[str, str]) -> bool:
    addresses = (row.get("ROAD_NM_ADDR", ""), row.get("LOTNO_ADDR", ""))
    if any(clean(a).startswith(CAPITAL_PREFIXES) for a in addresses if clean(a)):
        return True
    sgg = re.sub(r"\D", "", clean(row.get("SGG_CD", "")))
    return len(sgg) >= 2 and sgg[:2] in CAPITAL_SIDO_CODES


def schema_and_inventory(data_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    inventory: list[dict[str, object]] = []
    schemas: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for year in YEARS:
        root = year_dir(data_dir, year)
        for split_dir, split in SPLITS.items():
            for path in sorted((root / split_dir).glob("*.csv")):
                if "poi_master" in path.name:
                    continue
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    header = next(csv.reader(handle))
                    rows = sum(1 for _ in handle)
                table = "_".join(path.name.split("_")[:3])
                inventory.append(
                    {
                        "year": year,
                        "source_split": split,
                        "table": table,
                        "file": nfc(str(path.relative_to(data_dir))),
                        "bytes": path.stat().st_size,
                        "rows": rows,
                        "columns": len(header),
                        "used_for_common_dataset": "1"
                        if path.name.startswith(
                            ("tn_traveller_master", "tn_travel_", "tn_companion_info", "tn_visit_area_info")
                        )
                        else "0",
                        "schema": "|".join(header),
                    }
                )
                schemas[table][year] = header
    compatibility = {
        table: {
            "same_columns_and_order": len(by_year) == 2
            and by_year.get("2022") == by_year.get("2023"),
            "columns_2022": by_year.get("2022", []),
            "columns_2023": by_year.get("2023", []),
        }
        for table, by_year in schemas.items()
    }
    return inventory, compatibility


def find_poi_master(data_dir: Path) -> Path | None:
    preferred = sorted(data_dir.rglob("tn_poi_master_POIMaster_all.csv"))
    if preferred:
        return preferred[0]
    fallback = [p for p in data_dir.rglob("tn_poi_master*.csv") if p.parent.name == "TL_csv"]
    return sorted(fallback)[0] if fallback else None


def load_poi_master(path: Path | None, wanted: set[str]) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    if path is None:
        return {}, {"path": None, "rows_scanned": 0, "matched_ids": 0}
    found: dict[str, dict[str, str]] = {}
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            poi_id = clean(row.get("POI_ID"))
            if poi_id in wanted and poi_id not in found:
                found[poi_id] = {k: clean(v) for k, v in row.items()}
    return found, {
        "path": nfc(str(path)),
        "rows_scanned": rows,
        "referenced_ids": len(wanted),
        "matched_ids": len(found),
        "unmatched_ids": len(wanted - set(found)),
    }


def load_sources(data_dir: Path) -> tuple[dict[str, object], list[dict[str, str]]]:
    users: dict[tuple[str, str], dict[str, str]] = {}
    travels: dict[tuple[str, str], dict[str, str]] = {}
    companions: dict[tuple[str, str], set[str]] = defaultdict(set)
    visits: list[dict[str, str]] = []
    audit: dict[str, object] = {
        "raw_rows": defaultdict(lambda: defaultdict(dict)),
        "duplicate_rows": defaultdict(dict),
        "user_profile_conflicts": defaultdict(int),
        "travel_conflicts": defaultdict(int),
    }

    for year in YEARS:
        root = year_dir(data_dir, year)
        for split_dir, split in SPLITS.items():
            paths = {
                "users": table_path(root, split_dir, "tn_traveller_master"),
                "travels": table_path(root, split_dir, "tn_travel_"),
                "companions": table_path(root, split_dir, "tn_companion_info"),
                "visits": table_path(root, split_dir, "tn_visit_area_info"),
            }
            for name, path in paths.items():
                rows = read_csv(path)
                audit["raw_rows"][year][split][name] = len(rows)
                audit["duplicate_rows"][year][f"{split}_{name}"] = len(rows) - len(
                    {tuple(sorted(r.items())) for r in rows}
                )
                if name == "users":
                    for row in rows:
                        key = (year, row["TRAVELER_ID"])
                        if key in users and users[key] != row:
                            audit["user_profile_conflicts"][year] += 1
                        users.setdefault(key, row)
                elif name == "travels":
                    for row in rows:
                        key = (year, row["TRAVEL_ID"])
                        if key in travels and travels[key] != row:
                            audit["travel_conflicts"][year] += 1
                        travels.setdefault(key, row)
                elif name == "companions":
                    for row in rows:
                        companions[(year, row["TRAVEL_ID"])].add(row.get("REL_CD", ""))
                else:
                    for row in rows:
                        row["year"] = year
                        row["source_split"] = split
                        visits.append(row)

    audit["raw_rows"] = {y: dict(v) for y, v in audit["raw_rows"].items()}
    audit["duplicate_rows"] = {y: dict(v) for y, v in audit["duplicate_rows"].items()}
    audit["user_profile_conflicts"] = dict(audit["user_profile_conflicts"])
    audit["travel_conflicts"] = dict(audit["travel_conflicts"])
    audit["user_profiles"] = {y: sum(1 for yy, _ in users if yy == y) for y in YEARS}
    audit["travels"] = {y: sum(1 for yy, _ in travels if yy == y) for y in YEARS}

    enriched: list[dict[str, str]] = []
    missing_trip = Counter()
    missing_profile = Counter()
    seen_interaction_ids: set[str] = set()
    duplicate_interaction_ids = 0
    for row in visits:
        year = row["year"]
        travel = travels.get((year, row["TRAVEL_ID"]))
        traveler_id = travel.get("TRAVELER_ID", "") if travel else ""
        profile = users.get((year, traveler_id)) if traveler_id else None
        if travel is None:
            missing_trip[year] += 1
        if profile is None:
            missing_profile[year] += 1
        row["TRAVELER_ID"] = traveler_id
        row["user_id"] = f"{year}:{traveler_id}" if traveler_id else ""
        row["profile_exists"] = "1" if profile else "0"
        relations = sorted(x for x in companions.get((year, row["TRAVEL_ID"]), set()) if x)
        row["companion_relation_codes"] = ";".join(relations)
        row["couple_relation"] = "lover" if "8" in relations else "spouse" if "1" in relations else ""
        iid = f"{year}:{row['TRAVEL_ID']}:{row['VISIT_AREA_ID']}"
        if iid in seen_interaction_ids:
            duplicate_interaction_ids += 1
            iid = f"{iid}:{row['source_split']}"
        seen_interaction_ids.add(iid)
        row["interaction_id"] = iid
        enriched.append(row)

    audit["join_missing"] = {
        y: {
            "trip": missing_trip[y],
            "profile": missing_profile[y],
            "trip_pct": round(100 * missing_trip[y] / max(1, sum(1 for r in visits if r["year"] == y)), 6),
            "profile_pct": round(100 * missing_profile[y] / max(1, sum(1 for r in visits if r["year"] == y)), 6),
        }
        for y in YEARS
    }
    audit["duplicate_interaction_ids"] = duplicate_interaction_ids
    audit["_users"] = users
    return audit, enriched


def fill_from_master(row: dict[str, str], master: dict[str, dict[str, str]]) -> None:
    poi = master.get(row.get("POI_ID", ""))
    if not poi:
        return
    mapping = {
        "POI_NM": "POI_NM",
        "ROAD_NM_ADDR": "ROAD_NM_ADDR",
        "LOTNO_ADDR": "LOTNO_ADDR",
        "X_COORD": "X_COORD",
        "Y_COORD": "Y_COORD",
        "SGG_CD": "SGG_CD",
    }
    for dst, src in mapping.items():
        if not clean(row.get(dst)):
            row[dst] = clean(poi.get(src))


def resolve_places(visits: list[dict[str, str]], master: dict[str, dict[str, str]]) -> dict[str, object]:
    for row in visits:
        fill_from_master(row, master)

    poi_by_name_addr: dict[tuple[str, str], set[str]] = defaultdict(set)
    poi_by_name_coord: dict[tuple[str, str], set[str]] = defaultdict(set)
    poi_years: dict[str, set[str]] = defaultdict(set)
    poi_names_by_year: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in visits:
        poi = clean(row.get("POI_ID"))
        if not poi:
            continue
        names = {norm_text(row.get("VISIT_AREA_NM")), norm_text(row.get("POI_NM"))} - {""}
        addresses = {norm_text(row.get("ROAD_NM_ADDR")), norm_text(row.get("LOTNO_ADDR"))} - {""}
        coord = coord_key(row.get("X_COORD"), row.get("Y_COORD"))
        for name in names:
            for address in addresses:
                poi_by_name_addr[(name, address)].add(poi)
            if coord:
                poi_by_name_coord[(name, coord)].add(poi)
        poi_years[poi].add(row["year"])
        for name in names:
            poi_names_by_year[poi][row["year"]].add(name)

    unique_addr = {k: next(iter(v)) for k, v in poi_by_name_addr.items() if len(v) == 1}
    unique_coord = {k: next(iter(v)) for k, v in poi_by_name_coord.items() if len(v) == 1}
    ambiguous_addr_keys = sum(1 for v in poi_by_name_addr.values() if len(v) > 1)
    ambiguous_coord_keys = sum(1 for v in poi_by_name_coord.values() if len(v) > 1)

    method_counts = Counter()
    for row in visits:
        poi = clean(row.get("POI_ID"))
        method = "poi_id"
        confidence = "high"
        if not poi and row.get("VISIT_AREA_TYPE_CD") in RECOMMENDABLE_TYPES:
            names = [norm_text(row.get("VISIT_AREA_NM")), norm_text(row.get("POI_NM"))]
            addresses = [norm_text(row.get("ROAD_NM_ADDR")), norm_text(row.get("LOTNO_ADDR"))]
            matches = {
                unique_addr[(name, addr)]
                for name in names
                for addr in addresses
                if name and addr and (name, addr) in unique_addr
            }
            if len(matches) == 1:
                poi = next(iter(matches))
                method = "exact_name_address_to_poi"
                confidence = "high"
            else:
                coord = coord_key(row.get("X_COORD"), row.get("Y_COORD"))
                matches = {
                    unique_coord[(name, coord)]
                    for name in names
                    if name and coord and (name, coord) in unique_coord
                }
                if len(matches) == 1:
                    poi = next(iter(matches))
                    method = "exact_name_coordinate_to_poi"
                    confidence = "high"
        if poi:
            place_id = f"poi:{poi}"
        elif row.get("VISIT_AREA_TYPE_CD") not in RECOMMENDABLE_TYPES:
            place_id = stable_id("excluded_visit", row["interaction_id"])
            method, confidence = "excluded_visit", "not_applicable"
        else:
            name = norm_text(row.get("VISIT_AREA_NM") or row.get("POI_NM"))
            road = norm_text(row.get("ROAD_NM_ADDR"))
            lot = norm_text(row.get("LOTNO_ADDR"))
            coord = coord_key(row.get("X_COORD"), row.get("Y_COORD"))
            if name and road:
                place_id = stable_id("addr", name, road)
                method, confidence = "name_road_address", "medium"
            elif name and lot:
                place_id = stable_id("lot", name, lot)
                method, confidence = "name_lot_address", "medium"
            elif name and coord:
                place_id = stable_id("coord", name, coord)
                method, confidence = "name_coordinate", "medium"
            else:
                place_id = ""
                method, confidence = "unresolved", "low"
        row["place_id"] = place_id
        row["place_id_method"] = method
        row["place_id_confidence"] = confidence
        row["is_recommendable"] = "1" if row.get("VISIT_AREA_TYPE_CD") in RECOMMENDABLE_TYPES else "0"
        row["is_capital_area"] = "1" if is_capital_area(row) else "0"
        method_counts[method] += 1

    common_pois = {poi for poi, years in poi_years.items() if len(years) == 2}
    name_consistent = 0
    name_inconsistent = 0
    for poi in common_pois:
        a = poi_names_by_year[poi].get("2022", set())
        b = poi_names_by_year[poi].get("2023", set())
        if a & b:
            name_consistent += 1
        else:
            name_inconsistent += 1
    return {
        "method_counts": dict(method_counts),
        "ambiguous_name_address_keys": ambiguous_addr_keys,
        "ambiguous_name_coordinate_keys": ambiguous_coord_keys,
        "common_poi_ids": len(common_pois),
        "common_poi_name_consistent": name_consistent,
        "common_poi_name_inconsistent": name_inconsistent,
    }


def build_places(visits: list[dict[str, str]], master: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in visits:
        if row.get("place_id"):
            groups[row["place_id"]].append(row)
    places: list[dict[str, object]] = []
    for place_id, rows in groups.items():
        poi_id = place_id[4:] if place_id.startswith("poi:") else ""
        poi = master.get(poi_id, {})
        preferred_name = clean(poi.get("POI_NM")) or mode(
            [r.get("POI_NM", "") for r in rows] + [r.get("VISIT_AREA_NM", "") for r in rows]
        )
        road = clean(poi.get("ROAD_NM_ADDR")) or mode(r.get("ROAD_NM_ADDR", "") for r in rows)
        lot = clean(poi.get("LOTNO_ADDR")) or mode(r.get("LOTNO_ADDR", "") for r in rows)
        x = clean(poi.get("X_COORD")) or mode(r.get("X_COORD", "") for r in rows)
        y = clean(poi.get("Y_COORD")) or mode(r.get("Y_COORD", "") for r in rows)
        sgg = clean(poi.get("SGG_CD")) or mode(r.get("SGG_CD", "") for r in rows)
        row_for_geo = {"ROAD_NM_ADDR": road, "LOTNO_ADDR": lot, "SGG_CD": sgg}
        users = {r.get("user_id") for r in rows if r.get("user_id")}
        ratings = [to_int(r.get("DGSTFN")) for r in rows]
        ratings = [x for x in ratings if x is not None and 1 <= x <= 5]
        dates = [r.get("VISIT_START_YMD", "") for r in rows if r.get("VISIT_START_YMD")]
        place = {
            "place_id": place_id,
            "poi_id": poi_id,
            "place_name": preferred_name,
            "road_address": road,
            "lot_address": lot,
            "longitude": x,
            "latitude": y,
            "sgg_code": sgg,
            "visit_type_code": mode(r.get("VISIT_AREA_TYPE_CD", "") for r in rows),
            "poi_category_large": clean(poi.get("ASORT_LCLASDC")),
            "poi_category_medium": clean(poi.get("ASORT_MLSFCDC")),
            "poi_category_small": clean(poi.get("ASORT_SDASDC")),
            "identity_method": mode(r.get("place_id_method", "") for r in rows),
            "identity_confidence": mode(r.get("place_id_confidence", "") for r in rows),
            "source_years": ";".join(sorted({r["year"] for r in rows})),
            "first_visit_date": min(dates) if dates else "",
            "last_visit_date": max(dates) if dates else "",
            "visit_count": len(rows),
            "rated_visit_count": len(ratings),
            "unique_user_count": len(users),
            "mean_rating": f"{statistics.mean(ratings):.6f}" if ratings else "",
            "is_recommendable": "1" if any(r.get("VISIT_AREA_TYPE_CD") in RECOMMENDABLE_TYPES for r in rows) else "0",
            "is_capital_area": "1" if is_capital_area(row_for_geo) else "0",
        }
        place["is_model_candidate"] = (
            "1"
            if place["is_recommendable"] == "1"
            and place["is_capital_area"] == "1"
            and place["identity_confidence"] in {"high", "medium"}
            else "0"
        )
        places.append(place)
    return sorted(places, key=lambda x: str(x["place_id"]))


def prepare_interactions(visits: list[dict[str, str]], places: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    place_lookup = {str(p["place_id"]): p for p in places}
    drop = Counter()
    out: list[dict[str, object]] = []
    for row in visits:
        rating = to_int(row.get("DGSTFN"))
        if not row.get("user_id"):
            drop["missing_user_join"] += 1
            continue
        if row.get("profile_exists") != "1":
            drop["missing_user_profile"] += 1
            continue
        if rating is None or not 1 <= rating <= 5:
            drop["missing_or_invalid_rating"] += 1
            continue
        place = place_lookup.get(row.get("place_id", ""))
        if not place:
            drop["unresolved_place"] += 1
            continue
        if row.get("VISIT_AREA_TYPE_CD") not in RECOMMENDABLE_TYPES:
            drop["excluded_visit_type"] += 1
            continue
        if place["is_capital_area"] != "1":
            drop["outside_or_unknown_capital_area"] += 1
            continue
        if place["identity_confidence"] not in {"high", "medium"}:
            drop["low_identity_confidence"] += 1
            continue
        out.append(
            {
                "interaction_id": row["interaction_id"],
                "user_id": row["user_id"],
                "place_id": row["place_id"],
                "year": row["year"],
                "source_split": row["source_split"],
                "travel_id": row["TRAVEL_ID"],
                "visit_area_id": row["VISIT_AREA_ID"],
                "visit_date": row.get("VISIT_START_YMD", ""),
                "visit_order": row.get("VISIT_ORDER", ""),
                "rating": rating,
                "revisit_intention": to_int(row.get("REVISIT_INTENTION")) or "",
                "recommendation_intention": to_int(row.get("RCMDTN_INTENTION")) or "",
                "visit_type_code": row.get("VISIT_AREA_TYPE_CD", ""),
                "companion_relation_codes": row.get("companion_relation_codes", ""),
                "couple_relation": row.get("couple_relation", ""),
            }
        )

    pair_rows: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in out:
        pair_rows[(str(row["user_id"]), str(row["place_id"]))].append(row)
    user_pairs: dict[str, list[tuple[str, tuple[str, int, str]]]] = defaultdict(list)
    for (user, place), rows in pair_rows.items():
        max_row = max(
            rows,
            key=lambda r: (
                str(r["visit_date"]),
                to_int(r["visit_order"]) or -1,
                str(r["interaction_id"]),
            ),
        )
        tie = hashlib.sha1(f"{user}|{place}".encode()).hexdigest()
        user_pairs[user].append(
            (place, (str(max_row["visit_date"]), to_int(max_row["visit_order"]) or -1, tie))
        )

    pair_split: dict[tuple[str, str], str] = {}
    eligible_users = 0
    for user, pairs in user_pairs.items():
        pairs.sort(key=lambda x: x[1])
        if len(pairs) >= 5:
            eligible_users += 1
            for place, _ in pairs[:-2]:
                pair_split[(user, place)] = "train"
            pair_split[(user, pairs[-2][0])] = "validation"
            pair_split[(user, pairs[-1][0])] = "test"
        else:
            for place, _ in pairs:
                pair_split[(user, place)] = "train_only"

    train_items_by_year: dict[str, set[str]] = defaultdict(set)
    for row in out:
        split = pair_split[(str(row["user_id"]), str(row["place_id"]))]
        row["known_user_split"] = split
        row["temporal_split"] = "train" if row["year"] == "2022" else "test"
        if split in {"train", "train_only"}:
            train_items_by_year[str(row["year"])].add(str(row["place_id"]))
    for row in out:
        row["is_warm_item_in_known_user_train"] = (
            "1" if str(row["place_id"]) in train_items_by_year[str(row["year"])] else "0"
        )

    out.sort(key=lambda r: (str(r["year"]), str(r["user_id"]), str(r["visit_date"]), str(r["interaction_id"])))
    return out, {"drop_reasons_first_failure": dict(drop), "eligible_known_users": eligible_users}


def build_users(source_users: dict[tuple[str, str], dict[str, str]], interactions: list[dict[str, object]]) -> list[dict[str, object]]:
    counts = Counter(str(r["user_id"]) for r in interactions)
    users: list[dict[str, object]] = []
    for (year, traveler_id), src in sorted(source_users.items()):
        user_id = f"{year}:{traveler_id}"
        row: dict[str, object] = {
            "user_id": user_id,
            "year": year,
            "traveler_id": traveler_id,
            "gender": src.get("GENDER", ""),
            "age_group": src.get("AGE_GRP", ""),
            "residence_sgg_code": src.get("RESIDENCE_SGG_CD", ""),
            "interaction_count": counts[user_id],
        }
        for i in range(1, 4):
            row[f"preferred_sido_{i}"] = src.get(f"TRAVEL_LIKE_SIDO_{i}", "")
            row[f"preferred_sgg_{i}"] = src.get(f"TRAVEL_LIKE_SGG_{i}", "")
        for i in range(1, 9):
            row[f"travel_style_{i}"] = src.get(f"TRAVEL_STYL_{i}", "")
        users.append(row)
    return users


def sparsity(rows: list[dict[str, object]]) -> dict[str, object]:
    pairs = {(str(r["user_id"]), str(r["place_id"])) for r in rows}
    users = {u for u, _ in pairs}
    places = {p for _, p in pairs}
    denom = len(users) * len(places)
    by_place: dict[str, set[str]] = defaultdict(set)
    by_user: dict[str, set[str]] = defaultdict(set)
    for user, place in pairs:
        by_place[place].add(user)
        by_user[user].add(place)
    return {
        "rows": len(rows),
        "unique_pairs": len(pairs),
        "users": len(users),
        "places": len(places),
        "sparsity_pct": round(100 * (1 - len(pairs) / denom), 6) if denom else None,
        "single_user_places": sum(len(v) == 1 for v in by_place.values()),
        "single_user_place_pct": round(100 * sum(len(v) == 1 for v in by_place.values()) / max(1, len(places)), 6),
        "users_with_3plus_places": sum(len(v) >= 3 for v in by_user.values()),
        "users_with_5plus_places": sum(len(v) >= 5 for v in by_user.values()),
    }


def metrics_report(
    source_audit: dict[str, object],
    visits: list[dict[str, str]],
    places: list[dict[str, object]],
    interactions: list[dict[str, object]],
    place_audit: dict[str, object],
    prep_audit: dict[str, object],
    poi_audit: dict[str, object],
    compatibility: dict[str, object],
) -> dict[str, object]:
    quality = {}
    for year in YEARS:
        rows = [r for r in visits if r["year"] == year]
        quality[year] = {
            col: {
                "missing": sum(not clean(r.get(col)) for r in rows),
                "missing_pct": round(100 * sum(not clean(r.get(col)) for r in rows) / max(1, len(rows)), 6),
                "distinct_nonempty": len({clean(r.get(col)) for r in rows if clean(r.get(col))}),
            }
            for col in ("POI_ID", "ROAD_NM_ADDR", "X_COORD", "Y_COORD", "DGSTFN", "SGG_CD")
        }
        ratings = [to_int(r.get("DGSTFN")) for r in rows]
        ratings = [x for x in ratings if x is not None and 1 <= x <= 5]
        quality[year]["rating_distribution"] = dict(sorted(Counter(ratings).items()))
        quality[year]["rating_4_or_5_pct"] = round(
            100 * sum(x >= 4 for x in ratings) / max(1, len(ratings)), 6
        )

    yearly = {year: sparsity([r for r in interactions if r["year"] == year]) for year in YEARS}
    place_years: dict[str, set[str]] = defaultdict(set)
    for row in interactions:
        place_years[str(row["place_id"])].add(str(row["year"]))
    split_counts = Counter(str(r["known_user_split"]) for r in interactions)
    eval_rows = [r for r in interactions if r["known_user_split"] in {"validation", "test"}]
    warm_eval = sum(r["is_warm_item_in_known_user_train"] == "1" for r in eval_rows)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {k: v for k, v in source_audit.items() if not k.startswith("_")},
        "schema_compatibility": compatibility,
        "visit_quality": quality,
        "poi_master": poi_audit,
        "place_resolution": place_audit,
        "places": {
            "rows": len(places),
            "model_candidates": sum(p["is_model_candidate"] == "1" for p in places),
            "shared_model_interaction_places_across_years": sum(len(y) == 2 for y in place_years.values()),
        },
        "interactions": {
            "combined": sparsity(interactions),
            "by_year": yearly,
            "known_user_split_rows": dict(split_counts),
            "known_user_eval_warm_item_pct": round(100 * warm_eval / max(1, len(eval_rows)), 6),
            **prep_audit,
        },
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty output: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_spec(metrics: dict[str, object]) -> str:
    inter = metrics["interactions"]
    by_year = inter["by_year"]
    places = metrics["places"]
    joins = metrics["source"]["join_missing"]
    return f"""# DateMate 공통 데이터 명세

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

행 수: 2022 {by_year['2022']['rows']:,}, 2023 {by_year['2023']['rows']:,}, 합계 {inter['combined']['rows']:,}.

### `outputs/processed/places.csv`

`place_id`당 한 행. 이름·주소·좌표·방문 유형·POI 분류와 품질 플래그를 제공한다. `is_model_candidate=1`만 추천 후보로 사용한다. 전체 {places['rows']:,}개 중 후보는 {places['model_candidates']:,}개다. POI 분류는 연결된 행에서도 결측이 많으므로 이름 TF-IDF, 방문 유형, 지역, 좌표를 함께 사용한다.

### `outputs/processed/users.csv`

`user_id`당 한 행. 인구통계, 선호 지역 3개, `travel_style_1`~`8`을 보존한다. 여행 성향 축의 양 끝 의미는 공식 정의서 확인 전 임의로 이름을 붙이지 않는다.

### 감사 산출물

- `quality_report.json`: 결측·조인·희소도·연도 공통 장소·분할 통계.
- `source_inventory.csv`: 검사한 56개 TL/VL CSV의 파일명, 행/열 수, 헤더와 공통 데이터셋 사용 여부.

## 품질 판정

- 2022 방문→여행 연결 누락: {joins['2022']['trip']:,}행 ({joins['2022']['trip_pct']:.2f}%), 사용자 프로필 누락: {joins['2022']['profile']:,}행 ({joins['2022']['profile_pct']:.2f}%). 해당 행은 모델 상호작용에서 제외한다.
- 2023 방문→여행/사용자 연결 누락: {joins['2023']['trip']:,}/{joins['2023']['profile']:,}행.
- 모델 상호작용 희소도: 2022 {by_year['2022']['sparsity_pct']:.4f}%, 2023 {by_year['2023']['sparsity_pct']:.4f}%. 단일 사용자 장소 비중도 높아 CF는 가능하지만 보조 모델로 취급한다.
- 공통 모델 장소: {places['shared_model_interaction_places_across_years']:,}개. 연도 이동 평가는 전체와 공통 장소 하위집합을 따로 보고한다.
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
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs" / "processed")
    parser.add_argument("--spec-path", type=Path, default=project_root / "outputs" / "processed" / "data_spec.generated.md")
    parser.add_argument("--skip-poi-master", action="store_true", help="Faster diagnostic run without POI enrichment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Reject destinations inside source data before any writes.
    source = args.data_dir.resolve()
    for destination in (args.output_dir.resolve(), args.spec_path.resolve()):
        if destination == source or source in destination.parents:
            raise ValueError("Output destinations must be outside the read-only source data directory")
    inventory, compatibility = schema_and_inventory(args.data_dir)
    source_audit, visits = load_sources(args.data_dir)
    wanted_pois = {clean(r.get("POI_ID")) for r in visits if clean(r.get("POI_ID"))}
    master_path = None if args.skip_poi_master else find_poi_master(args.data_dir)
    master, poi_audit = load_poi_master(master_path, wanted_pois)
    place_audit = resolve_places(visits, master)
    places = build_places(visits, master)
    interactions, prep_audit = prepare_interactions(visits, places)
    users = build_users(source_audit.pop("_users"), interactions)
    metrics = metrics_report(
        source_audit,
        visits,
        places,
        interactions,
        place_audit,
        prep_audit,
        poi_audit,
        compatibility,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "interactions.csv", interactions)
    write_csv(args.output_dir / "places.csv", places)
    write_csv(args.output_dir / "users.csv", users)
    write_csv(args.output_dir / "source_inventory.csv", inventory)
    (args.output_dir / "quality_report.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.spec_path.parent.mkdir(parents=True, exist_ok=True)
    args.spec_path.write_text(make_spec(metrics), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "metrics": metrics["interactions"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
