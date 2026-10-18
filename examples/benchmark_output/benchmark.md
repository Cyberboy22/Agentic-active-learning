# Planner benchmark

Each planner was run on every objective across **20 seeds** at an equal experiment budget. For a given seed all planners share the same initial design, so the runs are paired and the differences reflect the decision rule alone.

**Gap closed** G = (best − init_best) / (reference − init_best): 0 means a planner never improved on its starting sample, 1 means it reached the optimum. **Exp. to 90%** is how many experiments were needed to close 90% of that gap, with the number of runs that got there at all in brackets.

## ackley

| planner | gap closed | final best | regret | exp. to 90% |
|---|---:|---:|---:|---:|
| random | 0.162 ± 0.140 | -16.14 ± 2.77 | 16.14 | – (0/20) |
| classical_bo | 0.523 ± 0.258 | -9.308 ± 5.26 | 9.308 | – (0/20) |
| heuristic | 0.507 ± 0.251 | -9.606 ± 5.18 | 9.606 | – (0/20) |

**Paired comparison vs `classical_bo` (gap closed):**

| planner | mean Δgap | wins | p (Wilcoxon) |
|---|---:|---:|---:|
| random | -0.361 | 4/20 | 0.000 |
| heuristic | -0.015 | 10/20 | 0.756 |

## concrete

| planner | gap closed | final best | regret | exp. to 90% |
|---|---:|---:|---:|---:|
| random | 0.584 ± 0.229 | 73.06 ± 3.25 | 4.686 | 44 (2/20) |
| classical_bo | 0.699 ± 0.239 | 74.17 ± 3.21 | 3.58 | 45 (6/20) |
| heuristic | 0.746 ± 0.276 | 75.53 ± 2.67 | 2.216 | 39 (9/20) |

**Paired comparison vs `classical_bo` (gap closed):**

| planner | mean Δgap | wins | p (Wilcoxon) |
|---|---:|---:|---:|
| random | -0.116 | 8/20 | 0.202 |
| heuristic | +0.047 | 12/20 | 0.445 |

## hartmann6

| planner | gap closed | final best | regret | exp. to 90% |
|---|---:|---:|---:|---:|
| random | 0.322 ± 0.183 | 1.709 ± 0.551 | 1.613 | – (0/20) |
| classical_bo | 0.783 ± 0.195 | 2.823 ± 0.459 | 0.4998 | 40 (7/20) |
| heuristic | 0.856 ± 0.152 | 2.969 ± 0.422 | 0.3532 | 38 (8/20) |

**Paired comparison vs `classical_bo` (gap closed):**

| planner | mean Δgap | wins | p (Wilcoxon) |
|---|---:|---:|---:|
| random | -0.461 | 3/20 | 0.000 |
| heuristic | +0.073 | 11/20 | 0.294 |

## Aggregate across objectives

| planner | mean gap closed | mean regret (normalised) |
|---|---:|---:|
| random | 0.356 | 0.644 |
| classical_bo | 0.668 | 0.332 |
| heuristic | 0.703 | 0.297 |

> A caveat worth stating: these are low-dimensional, cheap-to-evaluate benchmarks with a modest budget. They show whether a decision rule is sound, not how it behaves on a real campaign with expensive, noisy evaluations and far larger candidate spaces.
