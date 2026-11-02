"""Build the final, hash-checked model comparison from local result bundles.

The script reads aggregate metrics only. Row-level predictions and source data
remain under ``outputs/`` and are never copied into ``reports/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "global_mean",
    "item_mean",
    "popularity",
    "cbf_region",
    "item_knn_k40",
    "hybrid",
)
MODES = ("warm", "all")
RANKING_METRICS = (
    "recall_at_10",
    "ndcg_at_10",
    "recall_at_10_warm",
    "ndcg_at_10_warm",
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing result bundle: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def comparison_signature(bundle: dict) -> dict:
    """Fields that must match for a fair comparison within one candidate mode."""
    meta = bundle["meta"]
    metrics = bundle["metrics"]
    return {
        "year": meta["year"],
        "split": meta["split"],
        "git_commit": meta["git"]["commit"],
        "git_dirty": meta["git"]["dirty"],
        "protocol": meta["protocol"],
        "candidate_mode": meta["candidate_mode"],
        "candidate_sha256": meta["candidate_sha256"],
        "evaluation_pairs_sha256": meta["evaluation_pairs_sha256"],
        "positive_users_sha256": meta["positive_users_sha256"],
        "warm_positive_users_sha256": meta["warm_positive_users_sha256"],
        "data_sha256": meta["data_sha256"],
        "eval_users": metrics["eval_users"],
        "ranked_users": metrics["ranked_users"],
        "relevant_total": metrics["relevant_total"],
        "relevant_warm": metrics["relevant_warm"],
        "relevant_cold": metrics["relevant_cold"],
        "candidates": metrics["candidates"],
        "train_pairs": metrics["train_pairs"],
        "eval_pairs": metrics["eval_pairs"],
        "protocol_version": metrics["protocol_version"],
    }


def verify_common_context(bundles: list[dict], candidate_mode: str) -> dict:
    if not bundles:
        raise ValueError("no bundles to compare")
    expected = comparison_signature(bundles[0])
    if expected["git_dirty"]:
        raise ValueError("final comparison requires a clean Git checkout")
    if expected["candidate_mode"] != candidate_mode:
        raise ValueError(
            f"candidate mode mismatch: expected {candidate_mode}, "
            f"got {expected['candidate_mode']}"
        )
    for bundle in bundles[1:]:
        actual = comparison_signature(bundle)
        if actual != expected:
            different = [key for key in expected if actual.get(key) != expected[key]]
            raise ValueError(
                f"incomparable result bundles for {candidate_mode}: "
                + ", ".join(different)
            )
    return expected


def verify_hybrid_endpoint(cbf: dict, hybrid: dict, alpha: float) -> None:
    if alpha != 0:
        return
    for key in RANKING_METRICS:
        left = cbf["metrics"][key]
        right = hybrid["metrics"][key]
        if not math.isclose(left, right, rel_tol=0, abs_tol=1e-15):
            raise ValueError(
                f"alpha=0 Hybrid differs from CBF for {key}: {right} != {left}"
            )


def metric_row(bundle: dict) -> dict:
    meta = bundle["meta"]
    metrics = bundle["metrics"]
    return {
        "year": meta["year"],
        "split": meta["split"],
        "candidate_mode": meta["candidate_mode"],
        "model": meta["model"],
        "recall_at_10": metrics["recall_at_10"],
        "ndcg_at_10": metrics["ndcg_at_10"],
        "recall_at_10_warm": metrics["recall_at_10_warm"],
        "ndcg_at_10_warm": metrics["ndcg_at_10_warm"],
        "recommendable_user_ratio": metrics["recommendable_user_ratio"],
        "item_coverage": metrics["item_coverage"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "rating_supported_pairs": metrics["rating_supported_pairs"],
        "eval_users": metrics["eval_users"],
        "relevant_total": metrics["relevant_total"],
        "relevant_warm": metrics["relevant_warm"],
        "relevant_cold": metrics["relevant_cold"],
        "candidates": metrics["candidates"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def build_summary(experiment_root: Path, hybrid_root: Path, report_dir: Path) -> dict:
    alpha_2022 = float(load_json(hybrid_root / "2022_alpha_tuning.json")["best_alpha"])
    alpha_2023 = float(load_json(hybrid_root / "2023_alpha_tuning.json")["best_alpha"])

    rows_2022 = []
    contexts = {}
    for mode in MODES:
        bundles = [
            load_json(experiment_root / "known_user_2022" / mode / model / "test" / "metrics.json")
            for model in MODELS
        ]
        contexts[mode] = verify_common_context(bundles, mode)
        by_model = {bundle["meta"]["model"]: bundle for bundle in bundles}
        verify_hybrid_endpoint(by_model["cbf_region"], by_model["hybrid"], alpha_2022)
        rows_2022.extend(metric_row(bundle) for bundle in bundles)

    rows_2023 = []
    for mode in MODES:
        for split in ("validation", "test"):
            bundle = load_json(
                experiment_root / "known_user_2023" / mode / "hybrid" / split / "metrics.json"
            )
            if bundle["meta"]["git"]["dirty"]:
                raise ValueError("2023 Hybrid summary requires a clean Git checkout")
            row = metric_row(bundle)
            row["alpha"] = alpha_2023
            rows_2023.append(row)

    summary = {
        "protocol_version": "1.1",
        "source_commit": contexts["warm"]["git_commit"],
        "source_clean": True,
        "data_sha256": contexts["warm"]["data_sha256"],
        "verification": {
            "same_context_within_2022_candidate_mode": True,
            "hybrid_2022_alpha": alpha_2022,
            "hybrid_2022_equals_cbf_region": alpha_2022 == 0,
            "hybrid_2023_alpha": alpha_2023,
        },
        "contexts_2022": contexts,
        "results_2022_test": rows_2022,
        "results_2023_hybrid": rows_2023,
        "scope_notes": [
            "2022 is the primary within-year known-user comparison.",
            "2023 Hybrid is a separate within-year experiment, not the 2022-to-2023 external evaluation.",
            "The all catalog is fixed from full-period place metadata and is not a strict historical catalog.",
        ],
    }

    write_csv(report_dir / "final_model_comparison_2022.csv", rows_2022)
    write_csv(report_dir / "final_hybrid_2023.csv", rows_2023)
    (report_dir / "final_evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=ROOT / "outputs" / "experiments" / "v1_1",
    )
    parser.add_argument(
        "--hybrid-root", type=Path, default=ROOT / "outputs" / "hybrid_final"
    )
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    summary = build_summary(args.experiment_root, args.hybrid_root, args.report_dir)
    print(
        "verified final comparison: "
        f"commit={summary['source_commit'][:8]}, "
        f"2022 alpha={summary['verification']['hybrid_2022_alpha']:.2f}, "
        f"2023 alpha={summary['verification']['hybrid_2023_alpha']:.2f}"
    )


if __name__ == "__main__":
    main()
