#!/usr/bin/env python3
"""Synthetic conformance and trace-scaling benchmark for the paper.

This benchmark validates deterministic policy invariants over typed action
manifests. It does not evaluate language-model planning, prompt-injection
detection, production integrations, or end-to-end agent safety.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable


SEED = 20260907
SCENARIOS_PER_TEMPLATE = 20
POLICY_REPEATS = 2_000
TRACE_REPEATS = 5
TRACE_SIZES = (100, 1_000, 10_000, 100_000)
DECISIONS = ("ALLOW", "PREVIEW", "APPROVE", "DENY")
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Manifest:
    scenario_id: str
    template: str
    expected: str
    tenant_match: bool = True
    instruction_from_untrusted_data: bool = False
    sensitive_drilldown: bool = False
    snapshot_fresh: bool = True
    query_bounded: bool = True
    schema_conflict: bool = False
    audit_available: bool = True
    external_side_effect: bool = False
    bulk_count: int = 0
    bulk_limit: int = 100
    reversible_analysis: bool = False
    partitions: int = 1
    source_count: int = 1


def build_scenarios() -> list[Manifest]:
    templates = [
        Manifest("", "bounded_aggregate", "ALLOW"),
        Manifest("", "reversible_analysis", "PREVIEW", reversible_analysis=True),
        Manifest("", "external_communication", "APPROVE", external_side_effect=True),
        Manifest("", "cross_tenant_query", "DENY", tenant_match=False),
        Manifest("", "untrusted_record_instruction", "DENY", instruction_from_untrusted_data=True),
        Manifest("", "sensitive_drilldown", "APPROVE", sensitive_drilldown=True),
        Manifest("", "stale_snapshot", "PREVIEW", snapshot_fresh=False),
        Manifest("", "unbounded_scan", "DENY", query_bounded=False),
        Manifest("", "schema_source_conflict", "APPROVE", schema_conflict=True),
        Manifest("", "audit_unavailable", "DENY", audit_available=False),
        Manifest("", "bounded_bulk_mutation", "APPROVE", bulk_count=50),
        Manifest("", "over_limit_bulk_mutation", "DENY", bulk_count=250),
    ]
    rng = random.Random(SEED)
    scenarios: list[Manifest] = []
    for template in templates:
        for index in range(SCENARIOS_PER_TEMPLATE):
            scenarios.append(
                replace(
                    template,
                    scenario_id=f"{template.template}-{index + 1:02d}",
                    partitions=rng.randint(1, 256),
                    source_count=rng.randint(1, 8),
                )
            )
    rng.shuffle(scenarios)
    return scenarios


def allow_all(_: Manifest) -> str:
    return "ALLOW"


def static_guardrail(m: Manifest) -> str:
    """A deliberately coarse checklist without full manifest semantics."""
    if not m.tenant_match or m.instruction_from_untrusted_data or m.bulk_count > m.bulk_limit:
        return "DENY"
    if m.external_side_effect or m.bulk_count > 0:
        return "APPROVE"
    return "ALLOW"


def proposed_control_plane(m: Manifest) -> str:
    """Deterministic precedence: prohibitions, approval, preview, allow."""
    if (
        not m.tenant_match
        or m.instruction_from_untrusted_data
        or not m.query_bounded
        or not m.audit_available
        or m.bulk_count > m.bulk_limit
    ):
        return "DENY"
    if m.sensitive_drilldown or m.schema_conflict or m.external_side_effect or m.bulk_count > 0:
        return "APPROVE"
    if not m.snapshot_fresh or m.reversible_analysis:
        return "PREVIEW"
    return "ALLOW"


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile_value))
    return ordered[index]


def time_policy(policy: Callable[[Manifest], str], scenarios: list[Manifest]) -> tuple[float, float]:
    per_decision_us: list[float] = []
    for _ in range(POLICY_REPEATS):
        start = time.perf_counter_ns()
        for scenario in scenarios:
            policy(scenario)
        elapsed_ns = time.perf_counter_ns() - start
        per_decision_us.append(elapsed_ns / len(scenarios) / 1_000.0)
    return statistics.median(per_decision_us), percentile(per_decision_us, 0.95)


def policy_metrics(
    name: str, policy: Callable[[Manifest], str], scenarios: list[Manifest]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    for scenario in scenarios:
        predicted = policy(scenario)
        records.append(
            {
                "configuration": name,
                "scenario_id": scenario.scenario_id,
                "template": scenario.template,
                "expected": scenario.expected,
                "predicted": predicted,
                "correct": predicted == scenario.expected,
            }
        )

    expected_interventions = [r for r in records if r["expected"] in ("APPROVE", "DENY")]
    predicted_interventions = [r for r in records if r["predicted"] in ("APPROVE", "DENY")]
    true_interventions = [r for r in predicted_interventions if r["expected"] in ("APPROVE", "DENY")]
    unsafe_auto = [r for r in expected_interventions if r["predicted"] == "ALLOW"]
    latency_p50_us, latency_p95_us = time_policy(policy, scenarios)
    result = {
        "configuration": name,
        "scenarios": len(records),
        "decision_accuracy_pct": 100.0 * sum(bool(r["correct"]) for r in records) / len(records),
        "unsafe_auto_execution_pct": 100.0 * len(unsafe_auto) / len(expected_interventions),
        "intervention_precision_pct": (
            100.0 * len(true_interventions) / len(predicted_interventions)
            if predicted_interventions
            else 0.0
        ),
        "intervention_recall_pct": 100.0 * len(true_interventions) / len(expected_interventions),
        "policy_latency_p50_us": latency_p50_us,
        "policy_latency_p95_us": latency_p95_us,
    }
    return result, records


def event_for(index: int) -> dict[str, object]:
    event_types = (
        "catalog_lookup",
        "query_stage",
        "retrieval",
        "policy_check",
        "tool_observation",
    )
    critical_types = ("policy_denial", "approval", "scope_change", "external_side_effect", "failure")
    critical = index % 50 == 0
    event_type = critical_types[(index // 50) % len(critical_types)] if critical else event_types[index % len(event_types)]
    return {
        "sequence": index,
        "timestamp_ms": 1_800_000_000_000 + index,
        "event_type": event_type,
        "critical": critical,
        "tenant": f"tenant-{index % 7}",
        "source": f"source-{index % 13}",
        "partition": index % 257,
        "record_count": (index * 17) % 10_000,
        "digest": hashlib.sha256(f"event-payload-{index}".encode()).hexdigest(),
    }


def build_trace(event_count: int) -> dict[str, object]:
    chain = b""
    raw_bytes = 0
    critical_events: list[dict[str, object]] = []
    type_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    start = time.perf_counter_ns()
    for index in range(event_count):
        event = event_for(index)
        encoded = json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
        raw_bytes += len(encoded) + 1
        chain = hashlib.sha256(chain + encoded).digest()
        type_counts[str(event["event_type"])] += 1
        source_counts[str(event["source"])] += 1
        if bool(event["critical"]):
            critical_events.append(event)
    summary = {
        "event_count": event_count,
        "critical_events": critical_events,
        "event_type_counts": dict(sorted(type_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "root_hash": chain.hex(),
    }
    compact_bytes = len(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode())
    elapsed_s = (time.perf_counter_ns() - start) / 1_000_000_000.0
    expected_critical = (event_count - 1) // 50 + 1
    return {
        "events": event_count,
        "critical_events": len(critical_events),
        "critical_retention_pct": 100.0 * len(critical_events) / expected_critical,
        "raw_bytes": raw_bytes,
        "compact_bytes": compact_bytes,
        "storage_reduction_pct": 100.0 * (1.0 - compact_bytes / raw_bytes),
        "build_time_ms": elapsed_s * 1_000.0,
        "throughput_events_s": event_count / elapsed_s,
    }


def trace_metrics(event_count: int) -> dict[str, object]:
    runs = [build_trace(event_count) for _ in range(TRACE_REPEATS)]
    result = dict(runs[0])
    for field in ("build_time_ms", "throughput_events_s"):
        result[field] = statistics.median(float(run[field]) for run in runs)
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    scenarios = build_scenarios()
    policy_summaries: list[dict[str, object]] = []
    policy_records: list[dict[str, object]] = []
    for name, policy in (
        ("Allow-all baseline", allow_all),
        ("Static guardrail", static_guardrail),
        ("Proposed control plane", proposed_control_plane),
    ):
        summary, records = policy_metrics(name, policy, scenarios)
        policy_summaries.append(summary)
        policy_records.extend(records)

    trace_summaries = [trace_metrics(size) for size in TRACE_SIZES]
    summary = {
        "scope_note": (
            "Synthetic validation of deterministic policy conformance and trace compaction only; "
            "not an evaluation of language-model reasoning or production safety."
        ),
        "seed": SEED,
        "scenario_templates": 12,
        "scenarios_per_template": SCENARIOS_PER_TEMPLATE,
        "label_distribution": dict(sorted(Counter(s.expected for s in scenarios).items())),
        "policy_repeats": POLICY_REPEATS,
        "trace_repeats": TRACE_REPEATS,
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "processor": platform.processor() or "not reported",
        },
        "policy_results": policy_summaries,
        "trace_results": trace_summaries,
    }

    (ROOT / "results_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(ROOT / "policy_results.csv", policy_records)
    write_csv(ROOT / "trace_results.csv", trace_summaries)
    (ROOT / "scenario_manifests.json").write_text(
        json.dumps([asdict(scenario) for scenario in scenarios], indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
