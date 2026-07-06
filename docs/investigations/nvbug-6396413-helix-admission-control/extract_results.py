# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract measured NVBug 6396413 B300 run artifacts into results.csv."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

RUN_ID_PATTERN = re.compile(r"^(?P<slot>\d{2})-(?P<arm>[AB])-attempt-(?P<attempt>[1-3])$")
ACCURACY_PATTERN = re.compile(r"Evaluated accuracy:\s*([0-9]+(?:\.[0-9]+)?)")
TELEMETRY_MARKER = "NVBUG6396413_JSON "
ACCURACY_THRESHOLD = 91.997

CSV_FIELDS = (
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
)

SUMMARY_FIELDS = (
    "admission_decisions",
    "would_defer_count",
    "effective_deferred_requests",
    "effective_deferral_request_ms",
    "max_effective_deferral_ms",
    "blocked_poll_count",
    "blocked_poll_ms",
    "decision_ms",
)

INFRA_PATTERNS = (
    "FileNotFoundError",
    "No space left on device",
    "Input/output error",
    "Slurm job",
    "Xid",
    "uncorrectable ECC",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Experiment RESULT_ROOT")
    parser.add_argument("--output", type=Path, help="Defaults to ROOT/results.csv")
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")
    return parser.parse_args()


def read_status(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def parse_telemetry(log_text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        if TELEMETRY_MARKER not in line:
            continue
        payload = line.split(TELEMETRY_MARKER, 1)[1]
        try:
            record = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError(f"Malformed telemetry JSON: {payload!r}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Telemetry record must be an object: {record!r}")
        records.append(record)
    return records


def junit_duration(path: Path) -> float | None:
    if not path.exists():
        return None
    root = ET.parse(path).getroot()
    times = [
        float(case.attrib["time"]) for case in root.iter("testcase") if case.attrib.get("time")
    ]
    if len(times) == 1:
        return times[0]
    return None


def latest_accuracy(log_text: str) -> float | None:
    matches = ACCURACY_PATTERN.findall(log_text)
    return float(matches[-1]) if matches else None


def classify(log_text: str, pytest_rc: int, accuracy: float | None) -> str:
    if pytest_rc == 0:
        return "pass"
    if "The accuracy evaluation took too long to complete" in log_text:
        return "accuracy_evaluation_duration_failure"
    if accuracy is not None and accuracy < ACCURACY_THRESHOLD:
        return "accuracy_below_threshold"
    if re.search(r"out of memory|OutOfMemory", log_text, re.IGNORECASE):
        return "crash_oom"
    if re.search(
        r"failed to register|timed out.*register|service.*not.*ready|waiting for.*server.*timed",
        log_text,
        re.IGNORECASE,
    ):
        return "registration_timeout"
    if re.search(r"Hang detected|model.?forward.*hang", log_text, re.IGNORECASE):
        return "model_forward_hang"
    if pytest_rc in {124, 137} or re.search(r"Timeout|timed out", log_text, re.IGNORECASE):
        return "outer_timeout_unknown"
    return "unexpected_product_failure"


def median_summary(records: list[dict[str, Any]], field: str) -> str:
    values = [float(record[field]) for record in records if record.get(field) is not None]
    if not values:
        return ""
    return f"{statistics.median(values):.6f}"


def rank_consistency(records: list[dict[str, Any]]) -> str:
    by_rank: dict[int, str] = {}
    for record in records:
        rank = record.get("rank")
        digest = record.get("decision_digest")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or not isinstance(digest, str)
            or not digest
            or rank in by_rank
        ):
            return "unknown"
        by_rank[rank] = digest
    if len(by_rank) != 4:
        return "unknown"
    return "true" if len(set(by_rank.values())) == 1 else "false"


def extract_run(run_dir: Path, match: re.Match[str]) -> dict[str, str]:
    status_path = run_dir / "status.txt"
    log_path = run_dir / "pytest.log"
    if not status_path.exists() or not log_path.exists():
        raise ValueError(f"{run_dir.name}: missing status.txt or pytest.log")

    status = read_status(status_path)
    if status.get("run_id") != run_dir.name:
        raise ValueError(f"{run_dir.name}: status run_id does not match directory")
    if status.get("arm") != match.group("arm"):
        raise ValueError(f"{run_dir.name}: status arm does not match directory")
    if status.get("run_mode") != "primary":
        raise ValueError(f"{run_dir.name}: measured directory must use run_mode=primary")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    telemetry = parse_telemetry(log_text)
    summaries = [
        record
        for record in telemetry
        if record.get("event") == "summary" and record.get("final") is True
    ]
    accuracy = latest_accuracy(log_text)
    pytest_rc = int(status["pytest_rc"])
    cleanup_ok = (run_dir / "cleanup.ok").exists()
    protocol_ok = (run_dir / "protocol.ok").exists()
    infra = any(pattern.lower() in log_text.lower() for pattern in INFRA_PATTERNS)

    if not cleanup_ok:
        validity = "protocol_invalid"
        outcome = "not_run"
    elif infra:
        validity = "infra_excluded"
        outcome = "not_run"
    elif not protocol_ok:
        validity = "protocol_invalid"
        outcome = "not_run"
    else:
        validity = "valid"
        outcome = classify(log_text, pytest_rc, accuracy)

    completed_duration = junit_duration(run_dir / "junit.xml")
    duration = completed_duration if outcome in {"pass", "accuracy_below_threshold"} else None
    if duration is None:
        duration = float(status["duration_s"])

    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "slot": str(int(match.group("slot"))),
            "attempt": match.group("attempt"),
            "run_id": run_dir.name,
            "arm": match.group("arm"),
            "validity": validity,
            "cleanup_ok": str(cleanup_ok).lower(),
            "product_outcome": outcome,
            "pytest_rc": str(pytest_rc),
            "duration_s": f"{duration:.6f}",
            "accuracy_pct": "" if accuracy is None else f"{accuracy:.6f}",
            "rank_consistent": rank_consistency(summaries),
            "notes": "REVIEW_REQUIRED: verify key events, JUnit, telemetry, and stacks",
        }
    )
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
        row[field] = status.get(field, "")
    for field in SUMMARY_FIELDS:
        row[field] = median_summary(summaries, field)
    return row


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = (args.output or root / "results.csv").resolve()
    if output.exists() and not args.force:
        raise ValueError(
            f"Refusing to overwrite {output}; pass --force after preserving manual edits"
        )

    rows: list[dict[str, str]] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        match = RUN_ID_PATTERN.fullmatch(run_dir.name)
        if match:
            rows.append(extract_run(run_dir, match))
    if not rows:
        raise ValueError(f"No measured run directories found under {root}")

    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
