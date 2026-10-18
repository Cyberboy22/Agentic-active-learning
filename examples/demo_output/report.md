# Campaign report

An LLM/heuristic agent chose which experiments to run, round by round, to maximise concrete compressive strength under a fixed experiment budget.

## Result

- **Best strength found:** 80.59 MPa
- **Random baseline (same budget):** 73.96 MPa
- **Brute-force reference optimum (noise-free):** 77.75 MPa
- **Fraction of reference reached:** 103.7%
- **Experiments spent:** 50

> The reference optimum is the *noise-free* ground-truth maximum from a 200k-point brute-force scan. Measured values include experimental noise (σ = 1.5 MPa), so a single measurement can sit slightly above it. The point is the gap to the random baseline, found with the same number of experiments.


**Best mix design found:**

| variable | value | unit |
|---|---:|---|
| cement | 227.70 | kg/m^3 |
| slag | 251.66 | kg/m^3 |
| fly_ash | 200.00 | kg/m^3 |
| water | 215.52 | kg/m^3 |
| superplasticizer | 18.75 | kg/m^3 |
| coarse_agg | 817.29 | kg/m^3 |
| fine_agg | 620.15 | kg/m^3 |
| age | 365.00 | days |

## Per-round decisions (the explainable trail)

### Round 0 — best so far 71.65 MPa (budget used 13)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`gaussian_process`, kappa=2.0, batch=3
- **Rationale:** early phase: map the space, so explore with a high UCB kappa; using gaussian_process (n=10 observations); batch=3.
- **Selected experiments:**

  | predicted (MPa) | uncertainty ± | acq score | measured (MPa) |
  |---:|---:|---:|---:|
  | 65.84 | 10.14 | 86.124 | 60.13 |
  | 66.74 | 9.66 | 86.056 | 65.7 |
  | 64.77 | 10.59 | 85.937 | 36.58 |

### Round 1 — best so far 71.65 MPa (budget used 16)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`gaussian_process`, kappa=2.0, batch=3
- **Rationale:** early phase: map the space, so explore with a high UCB kappa; using gaussian_process (n=13 observations); batch=3.
- **Selected experiments:**

  | predicted (MPa) | uncertainty ± | acq score | measured (MPa) |
  |---:|---:|---:|---:|
  | 65.57 | 10.05 | 85.671 | 67.56 |
  | 65.4 | 10.1 | 85.609 | 65.68 |
  | 65.16 | 10.22 | 85.597 | 65.01 |

### Round 2 — best so far 75.39 MPa (budget used 19)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`gaussian_process`, kappa=2.5, batch=3
- **Rationale:** stalled for 2 rounds: one exploration burst (raise kappa) to escape the current basin; using gaussian_process (n=16 observations); batch=3.
- **Selected experiments:**

  | predicted (MPa) | uncertainty ± | acq score | measured (MPa) |
  |---:|---:|---:|---:|
  | 80.59 | 6.24 | 96.2 | 67.97 |
  | 78.84 | 5.28 | 92.029 | 71.74 |
  | 78.77 | 5.25 | 91.903 | 75.39 |

### Round 3 — best so far 75.39 MPa (budget used 22)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`gaussian_process`, kappa=2.0, batch=3
- **Rationale:** early phase: map the space, so explore with a high UCB kappa; using gaussian_process (n=19 observations); batch=3.
- **Selected experiments:**

  | predicted (MPa) | uncertainty ± | acq score | measured (MPa) |
  |---:|---:|---:|---:|
  | 65.53 | 10.63 | 86.796 | 63.23 |
  | 67.59 | 9.1 | 85.787 | 53.56 |
  | 66.17 | 9.56 | 85.283 | 54.88 |

### Round 4 — best so far 75.39 MPa (budget used 26)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`gaussian_process`, kappa=1.0, batch=4
- **Rationale:** mid phase: balance explore/exploit with a moderate UCB kappa; using gaussian_process (n=22 observations); batch=4.
- **Selected experiments:**

  | predicted (MPa) | uncertainty ± | acq score | measured (MPa) |
  |---:|---:|---:|---:|
  | 75.5 | 2.78 | 78.283 | 69.87 |
  | 75.59 | 2.69 | 78.28 | 71.11 |
  | 75.41 | 2.77 | 78.176 | 72.0 |
  | 75.61 | 2.56 | 78.17 | 72.93 |

### Round 5 — best so far 75.39 MPa (budget used 29)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`gaussian_process`, kappa=2.5, batch=3
- **Rationale:** stalled for 2 rounds: one exploration burst (raise kappa) to escape the current basin; using gaussian_process (n=26 observations); batch=3.
- **Selected experiments:**

  | predicted (MPa) | uncertainty ± | acq score | measured (MPa) |
  |---:|---:|---:|---:|
  | 65.92 | 11.69 | 95.136 | 31.89 |
  | 65.46 | 11.85 | 95.088 | 24.25 |
  | 71.47 | 9.02 | 94.013 | 23.29 |

### Round 6 — best so far 75.39 MPa (budget used 33)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`gaussian_process`, kappa=1.0, batch=4
- **Rationale:** mid phase: balance explore/exploit with a moderate UCB kappa; using gaussian_process (n=29 observations); batch=4.
- **Selected experiments:**

  | predicted (MPa) | uncertainty ± | acq score | measured (MPa) |
  |---:|---:|---:|---:|
  | 74.88 | 5.73 | 80.608 | 61.88 |
  | 75.51 | 4.33 | 79.842 | 61.54 |
  | 74.62 | 4.02 | 78.637 | 70.07 |
  | 74.54 | 3.62 | 78.161 | 73.16 |

### Round 7 — best so far 75.39 MPa (budget used 36)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`gaussian_process`, kappa=2.5, batch=3
- **Rationale:** stalled for 4 rounds: one exploration burst (raise kappa) to escape the current basin; using gaussian_process (n=33 observations); batch=3.
- **Selected experiments:**

  | predicted (MPa) | uncertainty ± | acq score | measured (MPa) |
  |---:|---:|---:|---:|
  | 58.75 | 12.06 | 88.907 | 56.99 |
  | 51.86 | 14.14 | 87.203 | 51.99 |
  | 50.55 | 14.1 | 85.809 | 59.91 |

### Round 8 — best so far 75.39 MPa (budget used 40)
- **Decision [heuristic]:** acquisition=`ei`, surrogate=`gaussian_process`, kappa=0.0, batch=4
- **Rationale:** late phase: refine the best region with Expected Improvement; using gaussian_process (n=36 observations); batch=4.
- **Selected experiments:**

  | predicted (MPa) | uncertainty ± | acq score | measured (MPa) |
  |---:|---:|---:|---:|
  | 73.87 | 2.6 | 0.445 | 73.86 |
  | 68.87 | 5.97 | 0.415 | 60.81 |
  | 72.85 | 3.14 | 0.37 | 69.72 |
  | 73.7 | 2.26 | 0.293 | 74.05 |

### Round 9 — best so far 75.39 MPa (budget used 43)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`random_forest`, kappa=2.5, batch=3
- **Rationale:** stalled for 6 rounds: one exploration burst (raise kappa) to escape the current basin; using random_forest (n=40 observations); batch=3.
- **Selected experiments:**

  | predicted (MPa) | uncertainty ± | acq score | measured (MPa) |
  |---:|---:|---:|---:|
  | 55.88 | 21.45 | 109.51 | 45.96 |
  | 55.8 | 19.66 | 104.944 | 65.37 |
  | 53.84 | 20.28 | 104.54 | 72.14 |

### Round 10 — best so far 75.39 MPa (budget used 47)
- **Decision [heuristic]:** acquisition=`ei`, surrogate=`random_forest`, kappa=0.0, batch=4
- **Rationale:** late phase: refine the best region with Expected Improvement; using random_forest (n=43 observations); batch=4.
- **Selected experiments:**

  | predicted (MPa) | uncertainty ± | acq score | measured (MPa) |
  |---:|---:|---:|---:|
  | 62.55 | 13.91 | 1.336 | 51.9 |
  | 64.59 | 12.3 | 1.283 | 65.11 |
  | 63.73 | 12.44 | 1.164 | 60.99 |
  | 64.62 | 11.67 | 1.122 | 61.57 |

### Round 11 — best so far 80.59 MPa (budget used 50)
- **Decision [heuristic]:** acquisition=`ucb`, surrogate=`random_forest`, kappa=2.5, batch=3
- **Rationale:** stalled for 8 rounds: one exploration burst (raise kappa) to escape the current basin; using random_forest (n=47 observations); batch=3.
- **Selected experiments:**

  | predicted (MPa) | uncertainty ± | acq score | measured (MPa) |
  |---:|---:|---:|---:|
  | 57.24 | 15.74 | 96.587 | 80.59 |
  | 62.92 | 13.23 | 96.005 | 76.39 |
  | 56.15 | 15.75 | 95.517 | 69.11 |
