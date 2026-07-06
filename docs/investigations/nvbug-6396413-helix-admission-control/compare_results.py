# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and summarize NVBug 6396413 B300 A/B result rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = {
    "slot",
    "attempt",
    "run_id",
    "arm",
    "validity",
    "cleanup_ok",
    "product_outcome",
    "host",
    "base_sha",
    "experiment_sha",
    "image_digest",
    "allocation_id",
    "wheel_sha256",
    "gpu_fingerprint",
    "model_fingerprint",
    "dataset_fingerprint",
    "test_id",
    "pytest_rc",
    "duration_s",
    "accuracy_pct",
    "admission_decisions",
    "would_defer_count",
    "effective_deferred_requests",
    "effective_deferral_request_ms",
    "max_effective_deferral_ms",
    "blocked_poll_count",
    "blocked_poll_ms",
    "decision_ms",
    "rank_consistent",
    "notes",
}

COMPLETED_OUTCOMES = {"pass", "accuracy_below_threshold"}
ACCURACY_THRESHOLD = 91.997
TRUE_VALUES = {"1", "true", "yes"}
VALIDITIES = {"valid", "infra_excluded", "protocol_invalid"}
PRODUCT_OUTCOMES = {
    "not_run",
    "pass",
    "accuracy_below_threshold",
    "accuracy_evaluation_duration_failure",
    "model_forward_hang",
    "transfer_stall",
    "registration_timeout",
    "crash_oom",
    "outer_timeout_unknown",
    "unexpected_product_failure",
}
TELEMETRY_FIELDS = (
    "admission_decisions",
    "would_defer_count",
    "effective_deferred_requests",
    "effective_deferral_request_ms",
    "max_effective_deferral_ms",
    "blocked_poll_count",
    "blocked_poll_ms",
    "decision_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--expected-order", default="A,B,B,A,B,A,A,B,A,B")
    parser.add_argument("--min-valid-per-arm", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def parse_bool(value: str, field: str, run_id: str) -> bool:
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{run_id}: {field} must be true or false, got {value!r}")


def parse_float(value: str, field: str, run_id: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{run_id}: {field} must be numeric, got {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{run_id}: {field} must be finite, got {value!r}")
    return parsed


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        return list(reader)


def one_value(rows: list[dict[str, str]], field: str) -> str:
    values = {row[field].strip() for row in rows}
    if len(values) != 1 or "" in values:
        raise ValueError(f"Valid rows have mixed or empty {field}: {sorted(values)}")
    return next(iter(values))


def validate_and_select(
    rows: list[dict[str, str]], expected_order: list[str], min_valid: int
) -> list[dict[str, str]]:
    if not rows:
        raise ValueError("results.csv has no rows")

    for row in rows:
        run_id = row["run_id"]
        validity = row["validity"].strip()
        outcome = row["product_outcome"].strip()
        if validity not in VALIDITIES:
            raise ValueError(f"{run_id}: unsupported validity {validity!r}")
        if outcome not in PRODUCT_OUTCOMES:
            raise ValueError(f"{run_id}: unsupported product_outcome {outcome!r}")
        if validity == "valid" and outcome == "not_run":
            raise ValueError(f"{run_id}: a valid row must have a product outcome")
        if row["arm"].strip() not in {"A", "B"}:
            raise ValueError(f"{run_id}: arm must be A or B")
        if not row["notes"].strip() or "REVIEW_REQUIRED" in row["notes"]:
            raise ValueError(
                f"{run_id}: review and replace the extractor's notes before comparison"
            )
        if not parse_bool(row["cleanup_ok"], "cleanup_ok", run_id):
            raise ValueError(f"{run_id}: cleanup failed; subsequent samples are contaminated")

    valid_rows = [row for row in rows if row["validity"].strip() == "valid"]
    selected_by_slot: dict[int, dict[str, str]] = {}
    for row in valid_rows:
        run_id = row["run_id"]
        slot = int(row["slot"])
        if slot < 1 or slot > len(expected_order):
            raise ValueError(f"{run_id}: slot {slot} is outside the expected measured range")
        if slot in selected_by_slot:
            raise ValueError(f"Multiple valid rows exist for measured slot {slot}")
        expected_arm = expected_order[slot - 1]
        if row["arm"].strip() != expected_arm:
            raise ValueError(
                f"{run_id}: slot {slot} expected arm {expected_arm}, got {row['arm']!r}"
            )
        rank_consistent = row["rank_consistent"].strip().lower()
        if rank_consistent not in TRUE_VALUES | {"0", "false", "no", "unknown"}:
            raise ValueError(f"{run_id}: rank_consistent must be true, false, or unknown")
        if rank_consistent in {"0", "false", "no"}:
            raise ValueError(f"{run_id}: admission fingerprints are rank-inconsistent")
        try:
            pytest_rc = int(row["pytest_rc"])
        except ValueError as error:
            raise ValueError(f"{run_id}: pytest_rc must be an integer") from error
        outcome = row["product_outcome"].strip()
        if outcome == "pass" and pytest_rc != 0:
            raise ValueError(f"{run_id}: pass requires pytest_rc=0")
        if outcome != "pass" and pytest_rc == 0:
            raise ValueError(f"{run_id}: non-pass outcome cannot have pytest_rc=0")
        if outcome in COMPLETED_OUTCOMES:
            duration = parse_float(row["duration_s"], "duration_s", run_id)
            accuracy = parse_float(row["accuracy_pct"], "accuracy_pct", run_id)
            if duration <= 0:
                raise ValueError(f"{run_id}: completed duration must be positive")
            if not 0 <= accuracy <= 100:
                raise ValueError(f"{run_id}: accuracy_pct must be between 0 and 100")
            if outcome == "pass" and accuracy < ACCURACY_THRESHOLD:
                raise ValueError(f"{run_id}: pass accuracy is below {ACCURACY_THRESHOLD}")
            if outcome == "accuracy_below_threshold" and accuracy >= ACCURACY_THRESHOLD:
                raise ValueError(f"{run_id}: below-threshold outcome has passing accuracy")
            for field in TELEMETRY_FIELDS:
                if not row[field].strip():
                    raise ValueError(f"{run_id}: completed row is missing {field}")
                if parse_float(row[field], field, run_id) < 0:
                    raise ValueError(f"{run_id}: {field} cannot be negative")
            if parse_float(row["admission_decisions"], "admission_decisions", run_id) <= 0:
                raise ValueError(f"{run_id}: expected at least one admission decision")
            if row["arm"].strip() == "B":
                for field in (
                    "effective_deferred_requests",
                    "effective_deferral_request_ms",
                    "max_effective_deferral_ms",
                    "blocked_poll_count",
                    "blocked_poll_ms",
                ):
                    if parse_float(row[field], field, run_id) != 0:
                        raise ValueError(f"{run_id}: arm B requires {field}=0")
        selected_by_slot[slot] = row

    missing_slots = [
        slot for slot in range(1, len(expected_order) + 1) if slot not in selected_by_slot
    ]
    if missing_slots:
        raise ValueError(f"Missing valid measured slots: {missing_slots}")

    selected = [selected_by_slot[slot] for slot in sorted(selected_by_slot)]
    for field in (
        "host",
        "base_sha",
        "experiment_sha",
        "image_digest",
        "allocation_id",
        "wheel_sha256",
        "gpu_fingerprint",
        "model_fingerprint",
        "dataset_fingerprint",
        "test_id",
    ):
        one_value(selected, field)

    counts = Counter(row["arm"].strip() for row in selected)
    for arm in ("A", "B"):
        if counts[arm] < min_valid:
            raise ValueError(
                f"Arm {arm} has {counts[arm]} valid rows; require at least {min_valid}"
            )
    return selected


def optional_floats(rows: list[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field, "").strip()
        if value:
            values.append(parse_float(value, field, row["run_id"]))
    return values


def arm_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    outcomes = Counter(row["product_outcome"].strip() for row in rows)
    completed = [
        parse_float(row["duration_s"], "duration_s", row["run_id"])
        for row in rows
        if row["product_outcome"].strip() in COMPLETED_OUTCOMES
    ]
    accuracy = optional_floats(
        [row for row in rows if row["product_outcome"].strip() in COMPLETED_OUTCOMES],
        "accuracy_pct",
    )
    result: dict[str, Any] = {
        "valid": len(rows),
        "outcomes": dict(sorted(outcomes.items())),
        "completed_count": len(completed),
    }
    if completed:
        result["completed_duration_s"] = {
            "min": min(completed),
            "median": statistics.median(completed),
            "mean": statistics.fmean(completed),
            "max": max(completed),
        }
    if accuracy:
        result["accuracy_pct"] = {
            "min": min(accuracy),
            "median": statistics.median(accuracy),
            "max": max(accuracy),
        }
    telemetry: dict[str, dict[str, float]] = {}
    for field in TELEMETRY_FIELDS:
        values = optional_floats(rows, field)
        if values:
            telemetry[field] = {
                "total": sum(values),
                "median": statistics.median(values),
                "max": max(values),
            }
    if telemetry:
        result["telemetry"] = telemetry
    return result


def build_summary(all_rows: list[dict[str, str]], selected: list[dict[str, str]]) -> dict[str, Any]:
    by_arm = {arm: [row for row in selected if row["arm"].strip() == arm] for arm in ("A", "B")}
    arm_results = {arm: arm_summary(rows) for arm, rows in by_arm.items()}
    median_delta_pct = None
    if all("completed_duration_s" in arm_results[arm] for arm in ("A", "B")):
        median_a = arm_results["A"]["completed_duration_s"]["median"]
        median_b = arm_results["B"]["completed_duration_s"]["median"]
        if median_b:
            median_delta_pct = (median_a - median_b) / median_b * 100.0

    excluded = Counter(
        row["validity"].strip() for row in all_rows if row["validity"].strip() != "valid"
    )
    rank_unknown = sum(row["rank_consistent"].strip().lower() == "unknown" for row in selected)
    outcome_difference = arm_results["A"]["outcomes"] != arm_results["B"]["outcomes"]
    any_product_failure = any(
        outcome != "pass" for row in selected for outcome in [row["product_outcome"].strip()]
    )
    extend = (
        any_product_failure
        or outcome_difference
        or rank_unknown > 0
        or (median_delta_pct is not None and abs(median_delta_pct) >= 5.0)
    )
    return {
        "invariants": {
            field: one_value(selected, field)
            for field in (
                "host",
                "base_sha",
                "experiment_sha",
                "image_digest",
                "allocation_id",
                "wheel_sha256",
                "gpu_fingerprint",
                "model_fingerprint",
                "dataset_fingerprint",
                "test_id",
            )
        },
        "arms": arm_results,
        "median_delta_a_minus_b_pct": median_delta_pct,
        "rank_consistency_unknown_rows": rank_unknown,
        "excluded_attempts": dict(sorted(excluded.items())),
        "recommendation": (
            "Extend to at least 10 valid runs per arm; screening criteria were triggered."
            if extend
            else "Five-per-arm screen shows no trigger, but it is not closure evidence."
        ),
    }


def format_number(value: Any) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# NVBug 6396413 B300 A/B Summary",
        "",
        "| Arm | Valid | Pass | Pass rate | Other outcomes | Completed median (s) | "
        "Completed mean (s) | Accuracy median (%) |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for arm in ("A", "B"):
        data = summary["arms"][arm]
        outcomes = data["outcomes"]
        other = (
            ", ".join(f"{name}={count}" for name, count in outcomes.items() if name != "pass")
            or "none"
        )
        duration = data.get("completed_duration_s", {})
        accuracy = data.get("accuracy_pct", {})
        pass_count = outcomes.get("pass", 0)
        pass_rate = pass_count / data["valid"] * 100.0
        lines.append(
            f"| {arm} | {data['valid']} | {pass_count} | {pass_rate:.1f}% | {other} | "
            f"{format_number(duration.get('median'))} | {format_number(duration.get('mean'))} | "
            f"{format_number(accuracy.get('median'))} |"
        )
    lines.extend(
        [
            "",
            "| Arm | Would-defer median/run | Effective-deferred median/run | "
            "Deferral request-ms median/run | Max effective deferral (ms) | "
            "Blocked-poll ms median/run | Decision ms median/run |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in ("A", "B"):
        telemetry = summary["arms"][arm].get("telemetry", {})
        lines.append(
            f"| {arm} | {format_number(telemetry.get('would_defer_count', {}).get('median'))} | "
            f"{format_number(telemetry.get('effective_deferred_requests', {}).get('median'))} | "
            f"{format_number(telemetry.get('effective_deferral_request_ms', {}).get('median'))} | "
            f"{format_number(telemetry.get('max_effective_deferral_ms', {}).get('max'))} | "
            f"{format_number(telemetry.get('blocked_poll_ms', {}).get('median'))} | "
            f"{format_number(telemetry.get('decision_ms', {}).get('median'))} |"
        )
    lines.extend(
        [
            "",
            f"Median delta `(A-B)/B`: {format_number(summary['median_delta_a_minus_b_pct'])}%",
            "",
            f"Rows with unknown rank consistency: {summary['rank_consistency_unknown_rows']}",
            "",
            f"Recommendation: {summary['recommendation']}",
            "",
            "This is B300 screening evidence, not a direct estimate of the historical B200 failure rate.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    expected_order = [arm.strip() for arm in args.expected_order.split(",") if arm.strip()]
    if not expected_order or any(arm not in {"A", "B"} for arm in expected_order):
        raise ValueError("--expected-order must be a comma-separated sequence of A and B")

    rows = read_rows(args.results_csv)
    selected = validate_and_select(rows, expected_order, args.min_valid_per_arm)
    summary = build_summary(rows, selected)
    markdown = to_markdown(summary)

    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    if args.json_output:
        args.json_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
