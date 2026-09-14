# Synthetic Control-Plane Validation

This package accompanies the paper *Trustworthy Enterprise AI Agents for Big
Data Infrastructures*. It provides a reproducible, Python-standard-library
benchmark for four deliberately narrow claims:

1. a hybrid score-and-override implementation conforms to sixteen
   pre-specified policy classes over typed action manifests;
2. its decisions remain stable under bounded weight and threshold changes and
   factor ablations;
3. strict schema validation rejects missing safety fields, incorrect types,
   and unknown fields before policy evaluation; and
4. a hierarchical trace can retain all designated critical events while
   compacting high-volume noncritical events into counts and a hash chain.

It does **not** evaluate an LLM, natural-language planning, prompt-injection
detection, human oversight, a production data platform, or end-to-end safety.
The reference risk model uses normalized factors for impact, privilege breadth,
data sensitivity, data scope, uncertainty, and irreversibility, with weights
`0.25, 0.15, 0.20, 0.15, 0.15, 0.10`. Score cutoffs are `0.20`, `0.45`, and
`0.70` for T0 through T3. Categorical policy floors and prohibitions override
the score. The proposed policy's perfect nominal conformance score remains an
implementation check against pre-specified synthetic classes, not a claim of
general safety.

## Reproduce

Run with Python 3.10 or newer; no third-party packages are required:

```bash
python control_plane_benchmark.py
```

The fixed seed is `20260907`. The script regenerates:

- `scenario_manifests.json`: 320 typed manifests across 16 balanced templates;
- `policy_results.csv`: per-scenario outputs for three configurations;
- `robustness_results.csv`: weight, threshold, and ablation results;
- `schema_mutation_results.csv`: decisions for valid and malformed payloads;
- `trace_results.csv`: trace-scaling results at 100 to 100,000 events; and
- `results_summary.json`: aggregate metrics and runtime environment.

Timing results are machine-dependent. Decision labels are synthetic and should
be replaced or supplemented with independently adjudicated domain policies,
realistic agent trajectories, and human-subject evaluation before making any
deployment claim.
