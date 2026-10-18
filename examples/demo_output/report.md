# Campaign report — concrete

Planner **heuristic** chose which experiments to run, round by round, to maximise the objective under a fixed budget of 50 experiments.

## Result

- **Best value found:** 77.81 MPa
- **random baseline (same budget):** 75.75 MPa  (**+2.053 MPa**)
- **Reference optimum (noise-free):** 77.75 MPa
- **Fraction of reference reached:** 100.1%
- **Experiments spent:** 50

> The reference optimum is the noise-free ground-truth maximum. Measurements include experimental noise, so a single measurement can sit slightly above it. The meaningful comparison is against the baseline at the same budget.

**Best design found:**

| variable | value | unit |
|---|---:|---|
| cement | 141.4 | kg/m^3 |
| slag | 129.1 | kg/m^3 |
| fly_ash | 186.7 | kg/m^3 |
| water | 143 | kg/m^3 |
| superplasticizer | 21.29 | kg/m^3 |
| coarse_agg | 977.2 | kg/m^3 |
| fine_agg | 860.8 | kg/m^3 |
| age | 296.8 | days |

## Provenance

- **Config hash:** `57632e0ae082`
- **Started / finished (UTC):** 2026-10-18T19:22:33+00:00 → 2026-10-18T19:22:35+00:00
- **Python:** 3.13.16 on Linux-6.18.44-fc-v139-x86_64-with-glibc2.39
- **Packages:** numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, anthropic 1.15.0

<details><summary>Full configuration</summary>

```json
{
  "objective": "concrete",
  "dim": 8,
  "planner": "heuristic",
  "budget": 50,
  "n_init": 10,
  "n_rounds": 12,
  "pool_size": 1500,
  "max_batch": 6,
  "seed": 0
}
```
</details>

## Tool usage

| tool | calls | total ms | failures |
|---|---:|---:|---:|
| `fit_surrogate` | 12 | 1282.1 | 0 |
| `predict` | 12 | 150.6 | 0 |
| `propose_candidates` | 12 | 22.8 | 0 |
| `run_experiment` | 13 | 0.7 | 0 |
| `score_candidates` | 12 | 1.1 | 0 |

## Per-round decisions (the explainable trail)

### Round 0 — best so far 71.65 MPa (budget used 13)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`gaussian_process`, kappa=2.0, batch=3
- **Rationale:** early phase: map the space, so explore with a high UCB kappa; using gaussian_process (n=10 observations); batch=3.
- **Selected experiments:**

  | predicted | uncertainty ± | acq score | measured |
  |---:|---:|---:|---:|
  | 65.84 | 10.14 | 86.12 | 60.13 |
  | 66.74 | 9.659 | 86.06 | 65.7 |
  | 64.77 | 10.59 | 85.94 | 36.58 |

### Round 1 — best so far 71.65 MPa (budget used 16)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`gaussian_process`, kappa=2.0, batch=3
- **Rationale:** early phase: map the space, so explore with a high UCB kappa; using gaussian_process (n=13 observations); batch=3.
- **Selected experiments:**

  | predicted | uncertainty ± | acq score | measured |
  |---:|---:|---:|---:|
  | 65.48 | 10.09 | 85.67 | 66.67 |
  | 66.24 | 9.679 | 85.6 | 54.78 |
  | 66.38 | 9.578 | 85.53 | 65.11 |

### Round 2 — best so far 72.4 MPa (budget used 19)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`gaussian_process`, kappa=2.5, batch=3
- **Rationale:** stalled for 2 rounds: one exploration burst (raise kappa) to escape the current basin; using gaussian_process (n=16 observations); batch=3.
- **Selected experiments:**

  | predicted | uncertainty ± | acq score | measured |
  |---:|---:|---:|---:|
  | 68.63 | 9.897 | 93.37 | 52.17 |
  | 62.01 | 12.24 | 92.61 | 56.99 |
  | 60.68 | 12.77 | 92.61 | 72.4 |

### Round 3 — best so far 75.46 MPa (budget used 22)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`gaussian_process`, kappa=2.0, batch=3
- **Rationale:** early phase: map the space, so explore with a high UCB kappa; using gaussian_process (n=19 observations); batch=3.
- **Selected experiments:**

  | predicted | uncertainty ± | acq score | measured |
  |---:|---:|---:|---:|
  | 66.94 | 10.78 | 88.5 | 72.83 |
  | 66.61 | 10.9 | 88.41 | 75.46 |
  | 67.37 | 10.48 | 88.34 | 71.05 |

### Round 4 — best so far 75.46 MPa (budget used 26)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`gaussian_process`, kappa=1.0, batch=4
- **Rationale:** mid phase: balance explore/exploit with a moderate UCB kappa; using gaussian_process (n=22 observations); batch=4.
- **Selected experiments:**

  | predicted | uncertainty ± | acq score | measured |
  |---:|---:|---:|---:|
  | 76.09 | 10.41 | 86.5 | 68.14 |
  | 76.2 | 9.244 | 85.44 | 52.97 |
  | 72.78 | 10.95 | 83.72 | 59.8 |
  | 75.42 | 7.668 | 83.09 | 52.28 |

### Round 5 — best so far 75.46 MPa (budget used 29)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`gaussian_process`, kappa=1.0, batch=3
- **Rationale:** mid phase: balance explore/exploit with a moderate UCB kappa; using gaussian_process (n=26 observations); batch=3.
- **Selected experiments:**

  | predicted | uncertainty ± | acq score | measured |
  |---:|---:|---:|---:|
  | 84.92 | 8.596 | 93.51 | 62.54 |
  | 83.57 | 8.713 | 92.28 | 47.95 |
  | 80.94 | 10.99 | 91.94 | 61.94 |

### Round 6 — best so far 75.46 MPa (budget used 33)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`gaussian_process`, kappa=2.5, batch=4
- **Rationale:** stalled for 2 rounds: one exploration burst (raise kappa) to escape the current basin; using gaussian_process (n=29 observations); batch=4.
- **Selected experiments:**

  | predicted | uncertainty ± | acq score | measured |
  |---:|---:|---:|---:|
  | 72.42 | 10.03 | 97.48 | 37.1 |
  | 71.52 | 10.03 | 96.61 | 37.1 |
  | 69.55 | 10.55 | 95.93 | 45.52 |
  | 72.06 | 8.618 | 93.61 | 34.9 |

### Round 7 — best so far 75.46 MPa (budget used 36)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`gaussian_process`, kappa=1.0, batch=3
- **Rationale:** mid phase: balance explore/exploit with a moderate UCB kappa; using gaussian_process (n=33 observations); batch=3.
- **Selected experiments:**

  | predicted | uncertainty ± | acq score | measured |
  |---:|---:|---:|---:|
  | 77.16 | 4.252 | 81.41 | 70.92 |
  | 75.91 | 3.662 | 79.57 | 69.32 |
  | 75.85 | 3.612 | 79.46 | 69.48 |

### Round 8 — best so far 75.46 MPa (budget used 40)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`gaussian_process`, kappa=2.5, batch=4
- **Rationale:** stalled for 4 rounds: one exploration burst (raise kappa) to escape the current basin; using gaussian_process (n=36 observations); batch=4.
- **Selected experiments:**

  | predicted | uncertainty ± | acq score | measured |
  |---:|---:|---:|---:|
  | 60.9 | 11.1 | 88.65 | 51.61 |
  | 66.7 | 8.661 | 88.35 | 56.45 |
  | 59.68 | 11.33 | 87.99 | 44.8 |
  | 68.49 | 7.776 | 87.93 | 71.49 |

### Round 9 — best so far 77.12 MPa (budget used 43)
- **Decision [heuristic]:** acquisition=`ei`, surrogate=`random_forest`, kappa=0.0, batch=3
- **Rationale:** late phase: refine the best region with Expected Improvement; using random_forest (n=40 observations); batch=3.
- **Selected experiments:**

  | predicted | uncertainty ± | acq score | measured |
  |---:|---:|---:|---:|
  | 70.09 | 10.47 | 2.026 | 72.86 |
  | 70.07 | 10.46 | 2.018 | 76.87 |
  | 69.61 | 10.85 | 2.017 | 77.12 |

### Round 10 — best so far 77.12 MPa (budget used 47)
- **Decision [heuristic]:** acquisition=`ei`, surrogate=`random_forest`, kappa=0.0, batch=4
- **Rationale:** late phase: refine the best region with Expected Improvement; using random_forest (n=43 observations); batch=4.
- **Selected experiments:**

  | predicted | uncertainty ± | acq score | measured |
  |---:|---:|---:|---:|
  | 72.31 | 10.17 | 2.094 | 74.74 |
  | 72.16 | 10.12 | 2.03 | 74.05 |
  | 72.74 | 9.552 | 2.01 | 73.71 |
  | 72.66 | 9.569 | 1.991 | 76.28 |

### Round 11 — best so far 77.81 MPa (budget used 50)
- **Decision [heuristic]:** acquisition=`ei`, surrogate=`random_forest`, kappa=0.0, batch=3
- **Rationale:** late phase: refine the best region with Expected Improvement; using random_forest (n=47 observations); batch=3.
- **Selected experiments:**

  | predicted | uncertainty ± | acq score | measured |
  |---:|---:|---:|---:|
  | 71.5 | 9.93 | 1.766 | 77.81 |
  | 72 | 9.352 | 1.714 | 76.48 |
  | 71.93 | 9.265 | 1.664 | 76.17 |
