# Synthetic Control-Plane Validation

This package accompanies the paper *Trustworthy Enterprise AI Agents for Big
Data Infrastructures*. It provides a reproducible, Python-standard-library
benchmark for two deliberately narrow claims:

1. a deterministic implementation conforms to twelve pre-specified policy
   invariants over typed action manifests; and
2. a hierarchical trace can retain all designated critical events while
   compacting high-volume noncritical events into counts and a hash chain.

It does **not** evaluate an LLM, natural-language planning, prompt-injection
detection, human oversight, a production data platform, or end-to-end safety.
The proposed policy's perfect conformance score is therefore an implementation
check against rules encoded in both the scenario labels and evaluator, not a
claim of general safety.

## Reproduce

Run with Python 3.10 or newer; no third-party packages are required:

```bash
python control_plane_benchmark.py
```

The fixed seed is `20260907`. The script regenerates:

- `scenario_manifests.json`: 240 typed manifests across 12 balanced templates;
- `policy_results.csv`: per-scenario outputs for three configurations;
- `trace_results.csv`: trace-scaling results at 100 to 100,000 events; and
- `results_summary.json`: aggregate metrics and runtime environment.

Timing results are machine-dependent. Decision labels are synthetic and should
be replaced or supplemented with independently adjudicated domain policies,
realistic agent trajectories, and human-subject evaluation before making any
deployment claim.
