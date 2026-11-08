# Screening strategy benchmark

12 targets x 5 strategies x 6 paired seeds. Metric: **recall** (recall of the true top-k at the cost budget).

## Per target

| strategy | cdk2#0 | cdk2#1 | cdk2#2 | cdk2#3 | cmet#0 | cmet#1 | cmet#2 | cmet#3 | egfr#0 | egfr#1 | egfr#2 | egfr#3 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `always_cofolding` | 0.067 ± 0.075 | 0.103 ± 0.069 | 0.050 ± 0.047 | 0.087 ± 0.050 | 0.077 ± 0.059 | 0.093 ± 0.088 | 0.087 ± 0.075 | 0.023 ± 0.027 | 0.033 ± 0.032 | 0.037 ± 0.056 | 0.070 ± 0.056 | 0.120 ± 0.028 | **0.071** |
| `always_docking` | 0.403 ± 0.086 | 0.673 ± 0.050 | 0.413 ± 0.085 | 0.583 ± 0.035 | 0.423 ± 0.082 | 0.603 ± 0.027 | 0.697 ± 0.055 | 0.200 ± 0.042 | 0.423 ± 0.024 | 0.393 ± 0.104 | 0.510 ± 0.028 | 0.580 ± 0.085 | **0.492** |
| `always_ml_score` | 0.010 ± 0.015 | 0.010 ± 0.015 | 0.003 ± 0.007 | 0.007 ± 0.009 | 0.017 ± 0.021 | 0.017 ± 0.021 | 0.017 ± 0.021 | 0.007 ± 0.009 | 0.007 ± 0.009 | 0.007 ± 0.009 | 0.007 ± 0.009 | 0.010 ± 0.010 | **0.010** |
| `bandit` | 0.057 ± 0.080 | 0.053 ± 0.075 | 0.040 ± 0.045 | 0.030 ± 0.050 | 0.063 ± 0.055 | 0.097 ± 0.080 | 0.113 ± 0.084 | 0.020 ± 0.037 | 0.050 ± 0.051 | 0.040 ± 0.040 | 0.057 ± 0.074 | 0.070 ± 0.122 | **0.058** |
| `cascade` | 0.340 ± 0.050 | 0.427 ± 0.039 | 0.370 ± 0.055 | 0.460 ± 0.063 | 0.343 ± 0.033 | 0.440 ± 0.091 | 0.517 ± 0.076 | 0.203 ± 0.056 | 0.317 ± 0.039 | 0.347 ± 0.050 | 0.383 ± 0.082 | 0.577 ± 0.066 | **0.394** |

## Does choosing the method per target beat committing to one?

The project's premise is that no single method generalises. If that is true, an adaptive strategy should beat the *best fixed method on each target* — and crucially, the best fixed method should not be the same one every time. If one method wins everywhere, the orchestration layer is unnecessary and this table is the honest place to find that out.

| target | best fixed method | its recall | cascade Δ (p) | bandit Δ (p) |
|---|---|---|---|---|
| cdk2#0 | `always_docking` | 0.403 | -0.063 (p=0.312) | **-0.347** (p=0.031) |
| cdk2#1 | `always_docking` | 0.673 | **-0.247** (p=0.031) | **-0.620** (p=0.031) |
| cdk2#2 | `always_docking` | 0.413 | -0.043 (p=0.375) | **-0.373** (p=0.031) |
| cdk2#3 | `always_docking` | 0.583 | **-0.123** (p=0.031) | **-0.553** (p=0.031) |
| cmet#0 | `always_docking` | 0.423 | -0.080 (p=0.156) | **-0.360** (p=0.031) |
| cmet#1 | `always_docking` | 0.603 | **-0.163** (p=0.031) | **-0.507** (p=0.031) |
| cmet#2 | `always_docking` | 0.697 | **-0.180** (p=0.031) | **-0.583** (p=0.031) |
| cmet#3 | `always_docking` | 0.200 | +0.003 (p=1.000) | **-0.180** (p=0.031) |
| egfr#0 | `always_docking` | 0.423 | **-0.107** (p=0.031) | **-0.373** (p=0.031) |
| egfr#1 | `always_docking` | 0.393 | -0.047 (p=0.312) | **-0.353** (p=0.031) |
| egfr#2 | `always_docking` | 0.510 | **-0.127** (p=0.031) | **-0.453** (p=0.031) |
| egfr#3 | `always_docking` | 0.580 | -0.003 (p=1.000) | **-0.510** (p=0.031) |

**Best fixed method differs across targets: no** (1 distinct winner(s) across 12 targets). On these targets one fixed method was best everywhere. That is evidence *against* needing an orchestration layer here, and it should be reported as such rather than explained away.

Δ is the paired mean difference against the best fixed method, with a Wilcoxon signed-rank p-value. Bold marks p < 0.05. With 6 seeds these intervals are wide; treat anything above p = 0.05 as 'not shown', not as 'slightly worse'.
