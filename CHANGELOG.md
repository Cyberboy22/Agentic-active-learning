# Changelog

## 0.3.0

Extended the framework from continuous optimisation to **virtual screening**:
a finite library instead of a box, a cost budget instead of a count of
experiments, and a planner that chooses *which method* to spend it with. The
box half is unchanged and its 18 tests still pass.

**Pool objectives.** `PoolObjective` makes the design space the rows of a fixed
candidate table. Every planner, surrogate, acquisition function, report and
provenance record works against it unchanged. Two things improve in the
discrete setting: the optimum is known exactly rather than Monte-Carlo
estimated, and "refine around the best result" becomes analogue search in the
catalogue rather than a Gaussian nudge in coordinate space.

**Real chemistry, offline.** `VirtualScreeningObjective` builds EGFR, CDK2 and
c-Met targets from published ligand sets and the NCI library, both bundled with
RDKit. Affinity comes from an explicitly documented structural-and-property
model — it is not a docking score and not a prediction — except for 24 c-Met
ligands with measured free energies, which are used directly.
`from_score_table` swaps in DOCKSTRING, a lab's own Vina output or assay data
in one call.

**A cost ladder instead of one oracle.** Five rungs from near-free retrieval to
a wet-lab assay, each with a price and a fidelity, where fidelity is the
correlation with truth and is **per target**: the structure-based rungs are
drawn from overlapping ranges independently, so the better method genuinely
differs by target. Budget is cost; measurements carry their fidelity, and
surrogates are fitted with weights accordingly.

**Screening metrics.** Shortlist recall (the headline), enrichment factor,
BEDROC, coverage recall, and cost to reach a given recall.

**New planners.** `cascade` (cheap triage, then confirm — the pipeline the
literature describes), `bandit` (UCB1 over the evaluator menu, the principled
non-LLM comparator), and `always_<rung>` fixed-method baselines.

**Diversity-aware batching.** Greedy max-min selection trading acquisition
score against separation, because the top-k by acquisition in a library tends
to be k near-identical analogues.

**The LLM planner, made testable.** The tool menu is now rendered from the live
registry rather than hard-coded in the prompt — the claim the README already
made. Prompts and replies are cached to disk, so campaigns replay exactly and
for free, which is what lets the LLM planner be benchmarked at all. A
`state_mode` switch corrupts or withholds the campaign state for the ablation.

**A human checkpoint.** Reviewers may approve, veto or reorder a batch before
any budget is spent; every override is logged with a reason and vetoed
compounds are not proposed again. The rule-based reviewer uses RDKit's PAINS
alerts.

**Checkpoint and resume for screening campaigns**, with a test asserting a
campaign split across wall-time slots is identical to an uninterrupted one, and
refusal to resume into a different configuration. `slurm/array_campaign.sbatch`
runs it as an array job.

**What the experiments found.** Method selection returned a clean negative at
its default operating point: committing to docking beat every adaptive strategy
on all twelve landscapes, including the seven where co-folding is the more
accurate model. The follow-up (`cost_sensitivity.py`) explains it and partly
reverses it — recall of a top-k shortlist rewards coverage, and an eight-times
pricier method measures an eighth as many compounds, so accuracy cannot get a
vote until the budget is large enough to cover the shortlist. Past that point
the winner flips to co-folding in five of eight settings. A side finding worth
keeping: a noisy evaluator's recall *decreases* as its budget grows, because
every extra measurement is another chance for a mediocre compound to draw a
lucky reading and displace a real hit.

**Four experiments that can fail.** Method selection against the best fixed
method per target; whether the LLM planner is actually reading the state;
whether diversity batching helps; and whether the expensive method can ever pay
for itself. All paired by seed, with Wilcoxon tests and
bootstrap confidence intervals.

**A 3D viewer** with a rigid-body docking engine that runs in the browser —
Vina-style terms on precomputed grids, searched by coarse scan plus annealing
and local descent — validated by redocking the imatinib crystal pose in ABL1
(about 0.3 Å in eight of ten seeds, a secondary pocket in the other two).

**Packaging and CI.** `pyproject.toml`, GitHub Actions running lint and both
test suites on 3.10 and 3.12, and `ensemble_mlp` for fingerprint-scale
features. Tests grew from 18 to 39.

### Three bugs worth recording

Each would have produced plausible-looking nonsense, and each now has a
regression test.

- **Recall rewarded measuring, not recognising.** A compound counted as found
  if a confirming evaluator had touched it, regardless of what the measurement
  said — so blind measurement scored well and evaluator accuracy barely
  mattered. Recall is now computed over the shortlist the campaign would
  actually hand over.
- **Cheap triage destroyed the pool it was triaging for.** Candidate proposal
  excluded anything already measured at any fidelity, so every compound a
  retrieval pass ranked became permanently ineligible for the docking run it
  was ranked for. Exclusion is now per fidelity.
- **Per-target fidelities moved between processes.** They were keyed on
  Python's `hash()` of a string, which is salted per process. Now hashlib.

Also: the bandit's first reward function rewarded improvement per unit cost,
which a near-free method wins trivially, so it learned to spend whole campaigns
on a rung that cannot confirm anything. The reward is now progress in the best
*confirmed* value per unit cost.

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
