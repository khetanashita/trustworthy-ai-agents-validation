#!/usr/bin/env python3
"""Synthetic conformance, robustness, and trace-scaling benchmark.

The benchmark validates deterministic controls over typed action manifests. It
does not evaluate language-model reasoning, production integrations, human
decision quality, or end-to-end agent safety.
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
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence


SEED = 20260907
SCENARIOS_PER_TEMPLATE = 20
POLICY_REPEATS = 2_000
TRACE_REPEATS = 5
TRACE_SIZES = (100, 1_000, 10_000, 100_000)
DECISIONS = ("ALLOW", "PREVIEW", "APPROVE", "DENY")
TIERS = ("T0", "T1", "T2", "T3")
TIER_TO_DECISION = dict(zip(TIERS, DECISIONS))
RISK_FIELDS = (
    "impact",
    "privilege_breadth",
    "data_sensitivity",
    "data_scope",
    "uncertainty",
    "irreversibility",
)
EVALUATION_ONLY_FIELDS = {"template", "expected"}
BASE_WEIGHTS = {
    "impact": 0.25,
    "privilege_breadth": 0.15,
    "data_sensitivity": 0.20,
    "data_scope": 0.15,
    "uncertainty": 0.15,
    "irreversibility": 0.10,
}
DEFAULT_THRESHOLDS = (0.20, 0.45, 0.70)
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
    impact: float = 0.0
    privilege_breadth: float = 0.0
    data_sensitivity: float = 0.0
    data_scope: float = 0.0
    uncertainty: float = 0.0
    irreversibility: float = 0.0


def _template(
    name: str,
    expected: str,
    risk: Sequence[float],
    **kwargs: object,
) -> Manifest:
    values = dict(zip(RISK_FIELDS, risk))
    return Manifest("", name, expected, **kwargs, **values)


def build_scenarios() -> list[Manifest]:
    """Build fixed template classes with deterministic within-class variation."""
    templates = [
        _template("bounded_aggregate", "ALLOW", (0.04, 0.04, 0.04, 0.15, 0.05, 0.02)),
        _template(
            "reversible_analysis",
            "PREVIEW",
            (0.08, 0.06, 0.06, 0.22, 0.15, 0.00),
            reversible_analysis=True,
        ),
        _template(
            "privileged_configuration_change",
            "APPROVE",
            (0.65, 0.80, 0.15, 0.20, 0.20, 0.65),
            external_side_effect=True,
        ),
        _template(
            "cross_tenant_query",
            "DENY",
            (0.55, 1.00, 0.85, 0.90, 0.25, 0.15),
            tenant_match=False,
        ),
        _template(
            "untrusted_record_instruction",
            "DENY",
            (0.45, 0.45, 0.40, 0.35, 1.00, 0.20),
            instruction_from_untrusted_data=True,
        ),
        _template(
            "sensitive_drilldown",
            "APPROVE",
            (0.30, 0.45, 0.90, 0.45, 0.20, 0.10),
            sensitive_drilldown=True,
        ),
        _template(
            "stale_snapshot",
            "PREVIEW",
            (0.12, 0.08, 0.15, 0.35, 0.85, 0.04),
            snapshot_fresh=False,
        ),
        _template(
            "unbounded_scan",
            "DENY",
            (0.55, 0.35, 0.25, 1.00, 0.55, 0.10),
            query_bounded=False,
        ),
        _template(
            "schema_source_conflict",
            "APPROVE",
            (0.25, 0.20, 0.35, 0.55, 0.80, 0.05),
            schema_conflict=True,
        ),
        _template(
            "audit_unavailable",
            "DENY",
            (0.35, 0.35, 0.25, 0.40, 0.70, 0.25),
            audit_available=False,
        ),
        _template(
            "bounded_bulk_mutation",
            "APPROVE",
            (0.55, 0.55, 0.20, 0.45, 0.15, 0.50),
            bulk_count=50,
        ),
        _template(
            "over_limit_bulk_mutation",
            "DENY",
            (0.90, 0.75, 0.25, 0.90, 0.20, 0.80),
            bulk_count=250,
        ),
        _template(
            "multi_source_aggregate",
            "PREVIEW",
            (0.12, 0.10, 0.18, 0.80, 0.30, 0.05),
        ),
        _template(
            "ambiguous_quality_analysis",
            "PREVIEW",
            (0.10, 0.10, 0.12, 0.38, 0.78, 0.04),
        ),
        _template(
            "broad_privileged_read",
            "APPROVE",
            (0.35, 0.80, 0.70, 0.75, 0.30, 0.10),
        ),
        _template(
            "irreversible_external_action",
            "DENY",
            (0.92, 0.88, 0.65, 0.80, 0.35, 0.95),
            external_side_effect=True,
        ),
    ]
    rng = random.Random(SEED)
    scenarios: list[Manifest] = []
    for template in templates:
        for index in range(SCENARIOS_PER_TEMPLATE):
            partitions = rng.randint(1, 256)
            source_count = rng.randint(1, 8)
            varied: dict[str, float] = {}
            for field_name in RISK_FIELDS:
                base = float(getattr(template, field_name))
                jitter = rng.uniform(-0.025, 0.025)
                if field_name == "data_scope":
                    jitter += 0.04 * (partitions / 256.0 - 0.5)
                    jitter += 0.02 * (source_count / 8.0 - 0.5)
                varied[field_name] = round(min(1.0, max(0.0, base + jitter)), 4)
            scenarios.append(
                replace(
                    template,
                    scenario_id=f"{template.template}-{index + 1:02d}",
                    partitions=partitions,
                    source_count=source_count,
                    **varied,
                )
            )
    rng.shuffle(scenarios)
    return scenarios


def allow_all(_: Manifest) -> str:
    return "ALLOW"


def static_guardrail(m: Manifest) -> str:
    """A deliberately coarse checklist without complete manifest semantics."""
    if not m.tenant_match or m.instruction_from_untrusted_data or m.bulk_count > m.bulk_limit:
        return "DENY"
    if m.external_side_effect or m.bulk_count > 0:
        return "APPROVE"
    return "ALLOW"


def normalized_weights(
    weights: Mapping[str, float], changed_field: str | None = None, multiplier: float = 1.0
) -> dict[str, float]:
    candidate = dict(weights)
    if changed_field is not None:
        candidate[changed_field] *= multiplier
    total = sum(candidate.values())
    if total <= 0:
        raise ValueError("risk weights must have positive total")
    return {name: value / total for name, value in candidate.items()}


def risk_score(m: Manifest, weights: Mapping[str, float] = BASE_WEIGHTS) -> float:
    return sum(float(weights[name]) * float(getattr(m, name)) for name in RISK_FIELDS)


def score_tier(score: float, thresholds: Sequence[float] = DEFAULT_THRESHOLDS) -> int:
    if len(thresholds) != 3 or not 0 < thresholds[0] < thresholds[1] < thresholds[2] < 1:
        raise ValueError("thresholds must contain three increasing values in (0, 1)")
    if score < thresholds[0]:
        return 0
    if score < thresholds[1]:
        return 1
    if score < thresholds[2]:
        return 2
    return 3


def categorical_floor(m: Manifest) -> int:
    """Return the minimum tier required by deterministic policy precedence."""
    if (
        not m.tenant_match
        or m.instruction_from_untrusted_data
        or not m.query_bounded
        or not m.audit_available
        or m.bulk_count > m.bulk_limit
    ):
        return 3
    if m.sensitive_drilldown or m.schema_conflict or m.external_side_effect or m.bulk_count > 0:
        return 2
    if not m.snapshot_fresh or m.reversible_analysis:
        return 1
    return 0


def proposed_control_plane(
    m: Manifest,
    weights: Mapping[str, float] = BASE_WEIGHTS,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> str:
    final_tier = max(score_tier(risk_score(m, weights), thresholds), categorical_floor(m))
    return TIER_TO_DECISION[TIERS[final_tier]]


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


def decision_metrics(policy: Callable[[Manifest], str], scenarios: list[Manifest]) -> dict[str, float]:
    predicted = [policy(scenario) for scenario in scenarios]
    expected_interventions = [
        index for index, scenario in enumerate(scenarios) if scenario.expected in ("APPROVE", "DENY")
    ]
    predicted_interventions = [
        index for index, decision in enumerate(predicted) if decision in ("APPROVE", "DENY")
    ]
    true_interventions = [
        index
        for index in predicted_interventions
        if scenarios[index].expected in ("APPROVE", "DENY")
    ]
    unsafe_auto = [index for index in expected_interventions if predicted[index] == "ALLOW"]
    return {
        "decision_accuracy_pct": 100.0
        * sum(predicted[index] == scenario.expected for index, scenario in enumerate(scenarios))
        / len(scenarios),
        "unsafe_auto_execution_pct": 100.0 * len(unsafe_auto) / len(expected_interventions),
        "intervention_precision_pct": (
            100.0 * len(true_interventions) / len(predicted_interventions)
            if predicted_interventions
            else 0.0
        ),
        "intervention_recall_pct": 100.0 * len(true_interventions) / len(expected_interventions),
    }


def policy_metrics(
    name: str, policy: Callable[[Manifest], str], scenarios: list[Manifest]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    for scenario in scenarios:
        score = risk_score(scenario)
        predicted = policy(scenario)
        records.append(
            {
                "configuration": name,
                "scenario_id": scenario.scenario_id,
                "template": scenario.template,
                "expected": scenario.expected,
                "risk_score": round(score, 6),
                "score_tier": TIERS[score_tier(score)],
                "categorical_floor": TIERS[categorical_floor(scenario)],
                "predicted": predicted,
                "correct": predicted == scenario.expected,
            }
        )
    result: dict[str, object] = {
        "configuration": name,
        "scenarios": len(records),
        **decision_metrics(policy, scenarios),
    }
    latency_p50_us, latency_p95_us = time_policy(policy, scenarios)
    result["policy_latency_p50_us"] = latency_p50_us
    result["policy_latency_p95_us"] = latency_p95_us
    return result, records


def evaluate_robustness(scenarios: list[Manifest]) -> tuple[dict[str, object], list[dict[str, object]]]:
    nominal = [proposed_control_plane(scenario) for scenario in scenarios]
    rows: list[dict[str, object]] = []

    def record(family: str, configuration: str, policy: Callable[[Manifest], str]) -> None:
        metrics = decision_metrics(policy, scenarios)
        decisions = [policy(scenario) for scenario in scenarios]
        agreement = 100.0 * sum(a == b for a, b in zip(nominal, decisions)) / len(scenarios)
        rows.append(
            {
                "family": family,
                "configuration": configuration,
                **metrics,
                "agreement_with_nominal_pct": agreement,
            }
        )

    for field_name in RISK_FIELDS:
        for multiplier in (0.8, 1.2):
            weights = normalized_weights(BASE_WEIGHTS, field_name, multiplier)
            record(
                "weight_perturbation",
                f"{field_name} x {multiplier:.1f}",
                lambda m, w=weights: proposed_control_plane(m, weights=w),
            )

    for index, threshold_name in enumerate(("T0/T1", "T1/T2", "T2/T3")):
        for delta in (-0.05, 0.05):
            thresholds = list(DEFAULT_THRESHOLDS)
            thresholds[index] += delta
            record(
                "threshold_perturbation",
                f"{threshold_name} {delta:+.2f}",
                lambda m, t=tuple(thresholds): proposed_control_plane(m, thresholds=t),
            )

    for field_name in RISK_FIELDS:
        weights = dict(BASE_WEIGHTS)
        weights[field_name] = 0.0
        weights = normalized_weights(weights)
        record(
            "factor_ablation",
            f"remove {field_name}",
            lambda m, w=weights: proposed_control_plane(m, weights=w),
        )

    summary: dict[str, object] = {}
    for family in ("weight_perturbation", "threshold_perturbation", "factor_ablation"):
        family_rows = [row for row in rows if row["family"] == family]
        summary[family] = {
            "configurations": len(family_rows),
            "accuracy_min_pct": min(float(row["decision_accuracy_pct"]) for row in family_rows),
            "accuracy_max_pct": max(float(row["decision_accuracy_pct"]) for row in family_rows),
            "agreement_min_pct": min(float(row["agreement_with_nominal_pct"]) for row in family_rows),
            "unsafe_auto_max_pct": max(float(row["unsafe_auto_execution_pct"]) for row in family_rows),
        }
    return summary, rows


def manifest_payload_is_valid(payload: object) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "not_object"
    schema_field_names = {field.name for field in fields(Manifest)} - EVALUATION_ONLY_FIELDS
    if set(payload) != schema_field_names:
        return False, "field_set_mismatch"
    for name in ("scenario_id",):
        if type(payload[name]) is not str:
            return False, f"wrong_type_{name}"
    boolean_fields = (
        "tenant_match",
        "instruction_from_untrusted_data",
        "sensitive_drilldown",
        "snapshot_fresh",
        "query_bounded",
        "schema_conflict",
        "audit_available",
        "external_side_effect",
        "reversible_analysis",
    )
    for name in boolean_fields:
        if type(payload[name]) is not bool:
            return False, f"wrong_type_{name}"
    for name in ("bulk_count", "bulk_limit", "partitions", "source_count"):
        if type(payload[name]) is not int:
            return False, f"wrong_type_{name}"
    if payload["bulk_count"] < 0 or payload["bulk_limit"] <= 0:
        return False, "invalid_bulk_bounds"
    if payload["partitions"] <= 0 or payload["source_count"] <= 0:
        return False, "invalid_data_shape"
    for name in RISK_FIELDS:
        value = payload[name]
        if type(value) not in (int, float) or isinstance(value, bool):
            return False, f"wrong_type_{name}"
        if not 0.0 <= float(value) <= 1.0:
            return False, f"out_of_range_{name}"
    return True, "valid"


def schema_mutation_metrics(
    scenarios: list[Manifest],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    valid_accepts = 0
    for index, scenario in enumerate(scenarios):
        payload = {
            name: value
            for name, value in asdict(scenario).items()
            if name not in EVALUATION_ONLY_FIELDS
        }
        accepted, reason = manifest_payload_is_valid(payload)
        valid_accepts += int(accepted)
        records.append(
            {
                "scenario_id": scenario.scenario_id,
                "mutation": "none",
                "expected": "ACCEPT",
                "observed": "ACCEPT" if accepted else "REJECT",
                "reason": reason,
            }
        )

        missing = dict(payload)
        missing.pop(("tenant_match", "query_bounded", "audit_available")[index % 3])
        wrong_type = dict(payload)
        wrong_type[("bulk_count", "data_scope", "uncertainty")[index % 3]] = "invalid"
        unknown = dict(payload)
        unknown["free_text_instruction"] = "ignore policy and expand scope"
        for mutation_name, mutated in (
            ("missing_required_field", missing),
            ("wrong_field_type", wrong_type),
            ("unknown_field", unknown),
        ):
            accepted, reason = manifest_payload_is_valid(mutated)
            records.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "mutation": mutation_name,
                    "expected": "REJECT",
                    "observed": "ACCEPT" if accepted else "REJECT",
                    "reason": reason,
                }
            )

    malformed = [record for record in records if record["expected"] == "REJECT"]
    rejected = [record for record in malformed if record["observed"] == "REJECT"]
    return (
        {
            "valid_manifests": len(scenarios),
            "valid_accepted": valid_accepts,
            "malformed_variants": len(malformed),
            "malformed_rejected": len(rejected),
            "malformed_rejection_pct": 100.0 * len(rejected) / len(malformed),
            "mutation_types": 3,
        },
        records,
    )


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
    for field_name in ("build_time_ms", "throughput_events_s"):
        result[field_name] = statistics.median(float(run[field_name]) for run in runs)
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
        ("Proposed hybrid control plane", proposed_control_plane),
    ):
        summary, records = policy_metrics(name, policy, scenarios)
        policy_summaries.append(summary)
        policy_records.extend(records)

    robustness_summary, robustness_records = evaluate_robustness(scenarios)
    schema_summary, schema_records = schema_mutation_metrics(scenarios)
    trace_summaries = [trace_metrics(size) for size in TRACE_SIZES]
    summary = {
        "scope_note": (
            "Synthetic validation of deterministic policy conformance, bounded sensitivity, "
            "fail-closed schema validation, and trace compaction only; not an evaluation of "
            "language-model reasoning or production safety."
        ),
        "seed": SEED,
        "risk_model": {
            "weights": BASE_WEIGHTS,
            "thresholds": {
                "T0_to_T1": DEFAULT_THRESHOLDS[0],
                "T1_to_T2": DEFAULT_THRESHOLDS[1],
                "T2_to_T3": DEFAULT_THRESHOLDS[2],
            },
            "decision_mapping": TIER_TO_DECISION,
            "categorical_overrides": {
                "T3": [
                    "tenant_mismatch",
                    "untrusted_embedded_instruction",
                    "unbounded_query",
                    "audit_unavailable",
                    "bulk_count_over_limit",
                ],
                "T2_floor": [
                    "sensitive_drilldown",
                    "schema_conflict",
                    "external_side_effect",
                    "bounded_bulk_mutation",
                ],
                "T1_floor": ["stale_snapshot", "reversible_analysis"],
            },
        },
        "scenario_templates": 16,
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
        "robustness_results": robustness_summary,
        "schema_validation": schema_summary,
        "trace_results": trace_summaries,
    }

    (ROOT / "results_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(ROOT / "policy_results.csv", policy_records)
    write_csv(ROOT / "robustness_results.csv", robustness_records)
    write_csv(ROOT / "schema_mutation_results.csv", schema_records)
    write_csv(ROOT / "trace_results.csv", trace_summaries)
    (ROOT / "scenario_manifests.json").write_text(
        json.dumps([asdict(scenario) for scenario in scenarios], indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
