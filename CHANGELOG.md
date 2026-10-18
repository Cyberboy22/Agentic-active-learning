# Changelog

## 0.2.0

Turned a single-problem demo into a reusable framework, and added the evidence
to judge whether its planning actually helps.

**Pluggable objectives.** The experiment is now behind an `Objective` interface
(`var_names`, `bounds`, `evaluate`, optional `is_feasible`). Implementing those
is the entire integration surface — every planner, surrogate, acquisition
function and report works against a new problem unchanged. See
`examples/custom_objective.py`.

**More to compare on.** Added the standard optimisation benchmarks — Branin,
Hartmann6, Ackley, Rastrigin — alongside the original concrete-mix objective, so
a planner is judged across landscapes of different character rather than on one
hand-made problem.

**A real tool registry.** Any callable registers as a tool with a description and
parameter schema; the campaign performs all its work through it. Every
invocation is logged with duration and success, so the record shows what the
agent actually ran, not just what it decided.

**Planner ladder.** Unified `random`, `classical_bo` (GP + Expected Improvement,
fixed), `heuristic` (adaptive schedule) and `llm` (Claude plans each round)
behind one interface, so they are directly comparable. LLM output is clamped
into the legal range and falls back to the heuristic on any failure.

**Provenance and resumability.** Each run writes a manifest — config hash, seed,
library versions, git commit — and checkpoints after every round, including both
random generators' internal state. An interrupted campaign resumes bit-exactly;
there is a test asserting a resumed run equals an uninterrupted one.
`stop_after_rounds` lets a long campaign run in chunks under a wall-time limit.

**Benchmark harness with statistics.** Runs every planner against every objective
across many seeds. Paired by seed (all planners share an initial design), scored
with the sign-safe *gap closed* metric, and compared with a Wilcoxon signed-rank
test. Reports sample efficiency, not just the final value.

**Tests** grew from 6 to 18, covering determinism, resume equivalence, LLM output
clamping, tool-call logging and the paired-design property of the benchmark.

**Finding.** Running the new harness at 20 seeds instead of 10 dissolved the
adaptive planner's apparent advantage over classical Bayesian optimisation
(+0.165 → +0.047 gap closed on the concrete objective; p = 0.445). Both
model-based planners still beat random search decisively. The headline claim in
0.1.0 — that the agent beat its baseline — was true only against random search,
and the stronger reading would have been an artefact of too few seeds.

## 0.1.0

Initial release: an LLM/heuristic agent orchestrating surrogate models,
uncertainty and acquisition functions to choose which experiments to run next on
a concrete-mix objective, with per-round rationales and a convergence report.
