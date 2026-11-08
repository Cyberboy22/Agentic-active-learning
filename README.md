# Agentic Active Learning

**Infrastructure for autonomous scientific campaigns: an agent decides which
expensive experiments to run next, records why, and can prove the result is
reproducible — now including virtual screening, where the agent also decides
*which method* to spend the budget with.**

When every evaluation costs real time or money, the hard question is not *how
do I run an experiment* but *which experiment is worth running next*. In
structure-based drug discovery there is a second question stacked on top of it:
**what should I measure it with?** An embedding model ranks a compound for
almost nothing; docking costs real CPU; a co-folding model costs more again; an
assay costs days. None of those is the right answer everywhere, and something
has to choose.

This repository is a small, readable framework for both decisions.

```
                    ┌──────────────────────────────────────────────┐
 library / design ─▶│  fit surrogate  (mean + uncertainty)         │
                    │  planner decides: strategy AND method   ◀───── campaign state
                    │  propose candidates (global + analogues)     │
                    │  rank, then pick a diverse batch            │
                    │  human checkpoint: approve / veto           │
                    │  spend cost: measure with the chosen method │
                    │  checkpoint                                  │
                    └───────────────────┬──────────────────────────┘
                                        ▼ repeat
                recall · enrichment · cost ledger · decision log · manifest
```

Two campaign types share every component:

| | `run.py` | `run_screening.py` |
|---|---|---|
| design space | continuous box | a fixed library of real molecules |
| budget | count of experiments | **cost**, in compute units |
| oracle | one | a **ladder** of methods at different cost and fidelity |
| "refine near the best" | trust region | analogue search in the catalogue |
| success | gap closed toward the optimum | **recall of the true top-k**, enrichment, cost-to-recall |

---

## Quick start

```bash
pip install -e ".[chem]"          # rdkit comes with the chem extra

python run.py --planner heuristic --objective concrete          # box campaign
python run_screening.py --target egfr --planner cascade         # screening campaign
python tests/test_pipeline.py && python tests/test_screening.py # 39 tests
```

Everything runs offline. No API key, no network, no docking engine required.

---

## The screening side

### What is real here, and what is not

This matters more than any result below, so it goes first.

**Real.** The molecules. Actives are published ligand sets for EGFR, CDK2 and
c-Met that ship inside RDKit; the library bulk is the NCI collection. Morgan
fingerprint similarity is computed as it would be in practice. The c-Met set
carries 24 measured free energies, converted with the real thermodynamic
relation and used directly. The protein structure and crystallographic ligand
pose in the 3D viewer are real.

**Not real.** The affinity assigned to every other molecule. There is no docking
engine in this repository, so the objective defines an explicit, documented
affinity model — fingerprint similarity to known actives, plus hydrophobic,
hydrogen-bonding and ligand-efficiency terms, weighted per target. It produces a
landscape with the right *character* for comparing decision rules. **It is not a
prediction about any compound**, and no number out of this repository should be
read as one.

If you have real scores, the adapter is one call:

```python
from src.objectives.screening import from_score_table

objective = from_score_table("dockstring_scores.csv",
                             smiles_column="smiles", value_column="score",
                             higher_is_better=False)   # docking: more negative is better
```

That is the path to running the whole framework on DOCKSTRING's score matrix, a
lab's own Vina output, or in-house assay data. Nothing else changes.

### The evaluator ladder

```python
retrieval   cost 0.002   embedding dot product; ranks only, cannot confirm a hit
ml_score    cost 0.05    learned scoring function
docking     cost 1.0     pose plus score — the unit of cost
cofolding   cost 8.0     co-folding model; excellent on some targets, poor on others
assay       cost 400     ground truth
```

Fidelity is the correlation between what a method reports and the truth, and it
is **per target**. The two structure-based rungs are drawn from overlapping
ranges, independently — so on one target docking is far better than co-folding
and on the next it is the other way round, at eight times the price. That
calibration is the one choice that decides whether method selection is a real
question or a ranking to read off, and it is set to straddle in the spirit of
the co-folding benchmark that reports concordance with experiment ranging from
almost nothing to strong depending on the system.

### Success is recall, not a maximum

The headline metric is **shortlist recall**: hand a chemist the k compounds the
campaign rates highest *on confirmed evidence*, and count how many of the truly
best k are in that list.

This is deliberately harsher than it first appears, and the harshness is the
point. An earlier version counted a compound as found if a confirming evaluator
had merely *touched* it — which quietly rewarded blind measurement and made
evaluator accuracy nearly irrelevant. There is now a test
(`test_recall_requires_recognising_not_merely_measuring`) asserting that
measuring every true hit with a method that cannot recognise them scores zero.

Also reported: enrichment factor at 1% and 5%, BEDROC, coverage recall as a
diagnostic, and **cost to reach a given recall** — the axis the screening
literature reports improvements on.

---

## Does choosing the method per target beat committing to one?

This is the project's central claim, written as an experiment that can fail.

```bash
python experiments/method_selection.py --variants 4 --seeds 6
```

Three real targets is far too small a sample to ask whether a method
generalises, so the experiment also builds deterministic **synthetic landscape
variants** on the same real molecules — different hydrophobic optima, a
different balance between chemotype recognition and physical fit. They are
labelled as synthetic throughout; they say something about how a decision rule
behaves across a range of landscapes, not about chemistry.

### The result: the premise did not reproduce

**12 landscapes × 5 strategies × 6 paired seeds.** Mean recall of the true top-50
at a budget of 240 cost units over a ~1,200-compound library:

| strategy | mean recall |
|---|---:|
| `always_docking` | **0.492** |
| `cascade` | 0.394 |
| `always_cofolding` | 0.071 |
| `bandit` | 0.058 |
| `always_ml_score` | 0.010 |

**The best fixed method was `always_docking` on all twelve landscapes** —
including the seven where co-folding is the *more accurate* model. No adaptive
strategy beat it anywhere; `cascade` came closest and was statistically
indistinguishable on four landscapes and significantly worse on six.

On this benchmark, as built, an orchestration layer is not needed. Committing
to a single mid-cost method beats every alternative tried. That is the opposite
of what the project's premise predicts, and it is reported here rather than
tuned away.

**Why — and what it says about the measurement.** Recall of the true top-k
rewards *coverage*: to hand over 50 compounds you must have measured roughly 50
of them. Co-folding costs eight docking runs, so at a fixed budget it measures
an eighth as many compounds. A method that is more accurate per measurement but
gets eight times fewer of them cannot win a metric built on breadth, however
accurate it is. The cost ratio dominates, and accuracy barely gets a vote.

That is a property of the cost model and the metric as much as of the methods,
so `experiments/cost_sensitivity.py` tests it directly — varying the shortlist
size and the budget on the landscapes where co-folding is clearly the better
model, to find whether there is *any* setting where the expensive rung pays for
itself.

**And at a larger budget, the premise comes back.** On the two landscapes where
co-folding is clearly the better model, varying the budget and the shortlist
size flips the winner:

| landscape | docking fidelity | co-folding fidelity | budget | k | docking | co-folding | winner |
|---|---:|---:|---:|---:|---:|---:|---|
| egfr#1 | 0.61 | 0.96 | 240 | 10 | 0.633 | 0.167 | docking |
| egfr#1 | 0.61 | 0.96 | 240 | 50 | 0.393 | 0.087 | docking |
| egfr#1 | 0.61 | 0.96 | 960 | 10 | 0.367 | **1.000** | co-folding |
| egfr#1 | 0.61 | 0.96 | 960 | 50 | 0.347 | **0.527** | co-folding |
| cdk2#0 | 0.69 | 0.96 | 240 | 10 | 0.300 | **0.333** | co-folding |
| cdk2#0 | 0.69 | 0.96 | 240 | 50 | 0.400 | 0.173 | docking |
| cdk2#0 | 0.69 | 0.96 | 960 | 10 | 0.300 | **0.867** | co-folding |
| cdk2#0 | 0.69 | 0.96 | 960 | 50 | 0.313 | **0.500** | co-folding |

Co-folding wins five of eight settings. The pattern is clean: once the budget
is large enough that the expensive method can measure more compounds than the
shortlist needs, coverage stops being the binding constraint and accuracy
decides — decisively, in one case recovering the entire true top-10.

So the premise does hold. It just does not hold *across targets at a fixed
operating point*; it holds **across operating points**. Which method is right
depends on the budget and on how long a shortlist is wanted, at least as much
as on the target. That is still an argument for something choosing — and it is
a sharper, more testable argument than the one the project started with,
because it names the variables the choice depends on.

**A second finding, which was not what I went looking for.** Docking's recall
*falls* as its budget rises — 0.633 to 0.367 at k=10 on egfr#1, with four times
the compounds measured. More measurement makes it worse. This is selection
bias: with a noisy estimator, every extra compound measured is another chance
for a mediocre one to draw a lucky high reading and displace a genuine hit from
the shortlist. The more you measure with a weak method, the more the top of
your ranking fills with noise.

That has a direct practical reading for a screening campaign: past a point,
buying more cheap measurements actively degrades the shortlist, and the budget
should go to accuracy instead. It also suggests the shortlist should be built
from a model's posterior rather than from raw measurements, which this
framework does not currently do. Reproduce with
`python experiments/cost_sensitivity.py --seeds 3`.

**What this means for the real project.** Three honest readings, and the choice
between them needs real docking scores rather than this simulator:

1. The premise may hold in reality and not here, because this cost model is
   invented. If co-folding is three times the price of docking rather than
   eight, the arithmetic changes completely.
2. The premise may hold at a different operating point — early hit-finding
   where a handful of confirmed leads matters more than recall of a broad
   shortlist.
3. The premise may be weaker than the reading list suggests, and the sensible
   default really is "dock everything you can afford."

The useful thing this harness contributes is that the question is now *askable*
with a number attached, and that the answer it currently gives is not the
flattering one.

The harness reports paired differences against **the best fixed method on each
target**, with a Wilcoxon signed-rank test and a bootstrap confidence interval,
and states plainly whether the best fixed method was the same one everywhere. If
it was, the orchestration layer is unnecessary here and the report says so.

### Two more experiments that can fail

```bash
python experiments/llm_ablation.py --seeds 6       # needs a key once, then replays free
python experiments/diversity_ablation.py --seeds 8
```

**Is the LLM planner reading the state?** It runs the same campaigns three ways:
the planner sees the true state, a shuffled state (same fields, permuted
values), or no state at all. A model asked to plan will always return a
confident strategy and a fluent justification; if `real` does not beat
`shuffled`, those justifications are narration written after the fact. Every
prompt and reply is cached to disk, so repeat runs are free and exactly
reproducible — which is what makes running this over enough seeds affordable,
and why the LLM planner is no longer excluded from benchmarking for cost.

**Does diversity batching help?** Varies one number and nothing else. A
confidence interval spanning zero is reported as a result, not a failure.

---

## Planners

| planner | decides | why it exists |
|---|---|---|
| `random` | nothing | the floor |
| `classical_bo` | fixed GP + Expected Improvement | the baseline adaptivity must beat |
| `heuristic` | an adaptive schedule | explore early, refine late, break plateaus |
| `always_<rung>` | commits to one method | every paper on the reading list is one of these |
| `cascade` | cheap triage, then confirm | the DrugCLIP/AdaptiveFlow pipeline as a decision rule |
| `bandit` | UCB1 over the evaluator menu | the principled non-LLM way to choose a method |
| `llm` | Claude reads the state and chooses | the thing being tested |

`bandit` is there because "better than nothing" is not the bar. Before claiming
a language model is good at choosing methods, it is worth knowing what a
textbook bandit does with the same choice.

Its reward is also a cautionary tale left in the comments: rewarding *any*
improvement per unit cost is trivially won by a near-free method that produces a
tiny gain for a tiny price, and the bandit duly learned to spend entire
campaigns on a rung that cannot confirm anything. The reward is now progress in
the best **confirmed** value per unit cost. That was a reward-specification bug,
not a finding about bandits.

### The LLM planner

Three things are deliberate:

- **The menu is data.** The tool list in the prompt is rendered from the live
  `ToolRegistry`. Registering a lab's docking wrapper changes what the agent can
  choose without editing the planner — a claim an earlier version made in the
  README while hard-coding the menu in the prompt string.
- **Decisions are cached**, keyed by a hash of the prompt, so campaigns replay
  exactly and for free.
- **Output is clamped.** An unknown acquisition function, a thousand-compound
  batch or an invented evaluator degrades to a legal decision; any API or parse
  failure falls back to the deterministic planner.

---

## The human checkpoint

DrugCLIP reports roughly a 10% hit rate from its fully automated pipeline and
roughly 17.5% when human experts chose what to take forward. That gap is a
measurable target for an orchestration layer, and closing it needs a seam a
chemist can reach into — plus a record of what they did when they reached.

Before any budget is spent, the proposed batch passes a reviewer that may
approve, veto or reorder it. Every override is logged with a reason, and vetoed
compounds are not proposed again. `RuleBasedReviewer` stands in for expert
triage using RDKit's PAINS substructure alerts and property limits; it is a
caricature of a medicinal chemist and labelled as one. `CallbackReviewer` wraps
any function, including a real person.

---

## Reproducibility on a cluster

Every run writes a manifest: configuration, its hash, the library hash, the
seeds, library versions, the git commit, and the evaluator ladder with its
fidelities. Every round checkpoints every measurement, the cost ledger, the
reviewer's blocklist and both generators' internal state.

```bash
python run_screening.py --target egfr --out runs/egfr --stop-after-rounds 6
python run_screening.py --target egfr --out runs/egfr --resume
```

A campaign split across wall-time slots produces exactly the result an
uninterrupted run would have — `test_chunked_campaign_matches_an_uninterrupted_one`
asserts the ledgers, recall, evaluator sequence and per-round costs are
identical. Resuming into a different configuration is refused rather than
silently producing a mixture of two experiments. `slurm/array_campaign.sbatch`
runs the whole thing as an array job.

---

## The 3D viewer

```bash
python tools/build_viewer_data.py      # exports campaigns + MMFF conformers
# then open viewer/docking_bench.html
```

A campaign readout beside a rigid-body docking engine that runs in the browser:
Vina-style scoring terms evaluated on precomputed interaction grids, searched by
a coarse scan over cavity space followed by annealing and local descent.

It can be checked, which is the only reason to trust it. "Redock crystal ligand"
removes imatinib from its known position in ABL1 (PDB 1IEP) and asks the engine
to find it again; across ten seeds offline it recovered the crystallographic
pose to about 0.3 Å in eight runs and fell into a secondary pocket in two. The
viewer shows whichever you get.

Limits, stated rather than hidden: the ligand is rigid, so torsional
flexibility — most of what makes docking hard — is not searched; the receptor is
rigid too; desolvation and electrostatics are absent; scores are in arbitrary
units, not kcal/mol. The embedded protein is ABL1 while the campaigns target
EGFR, CDK2 and c-Met, because ABL1 is the kinase with a bundled, validated
docking set-up. All are kinases with similar ATP sites, so placing these
compounds is chemically meaningful — but a pose there is not a prediction about
the campaign's target.

---

## The original box benchmark, unchanged

The continuous-optimisation half of this repository is untouched, including the
finding it exists to report:

| planner | mean gap closed |
|---|---:|
| `random` | 0.356 |
| `classical_bo` | 0.668 |
| `heuristic` | **0.703** |

Model-based planning decisively beats random search (p < 0.001 on the harder
landscapes). The adaptive agent is **not** demonstrably better than textbook
Bayesian optimisation: at 10 seeds it looked like a clear winner (+0.165 on
concrete), and doubling to 20 seeds shrank that to +0.047 with p = 0.445. The
apparent advantage was mostly noise.

That experience is why this repository reports spread everywhere. None of the
four virtual-screening papers that motivated the screening half runs anything
twice: every comparison in them rests on a single enrichment factor, a single
hit rate, a single concordance index. The benchmark harness here pairs every
strategy by seed, tests differences with Wilcoxon, and attaches bootstrap
confidence intervals.

---

## Scaling limits

Stated plainly, because the benchmarks here are small:

- `GPSurrogate` is scikit-learn's GP: cubic in the number of observations and
  refitted from scratch each round. It is the right model for tens of points and
  will not survive ten thousand. Use `ensemble_mlp` past a few hundred.
- `EnsembleMLPSurrogate` is the one built for fingerprint-scale features —
  independently initialised networks with the spread of their predictions
  standing in for a posterior.
- Candidate scoring is dense over a sampled subset of the library. Libraries in
  the billions need the tranche-style pruning AdaptiveFlow describes before this
  loop is reached; nothing here does that yet.
- The browser docking engine handles one pocket and a rigid ligand. It is a
  demonstration, not a screening tool.

---

## Project layout

```
src/
  objective.py            box Objective + budget-counting Oracle + PoolObjective
  pool.py                 molecular libraries, featurisation, bundled real chemistry
  objectives/screening.py EGFR/CDK2/c-Met targets, affinity model, score-table adapter
  evaluators.py           the cost/fidelity ladder, cost ledger, multi-fidelity oracle
  metrics.py              shortlist recall, enrichment, BEDROC, cost-to-recall
  diversity.py            max-min batch selection
  surrogate.py            random forest / GP / ensemble MLP, all fidelity-weighted
  acquisition.py          UCB, EI, greedy, explore + cost-aware variants
  planners.py             random / classical BO / heuristic / cascade / bandit / LLM
  human.py                reviewer channel: PAINS alerts, property triage, overrides
  campaign.py             the box campaign loop
  screening_campaign.py   the pool-based, cost-budgeted campaign loop
  benchmark.py            box benchmark with paired statistics
  screening_benchmark.py  screening benchmark: Wilcoxon + bootstrap + method selection
experiments/              method selection, LLM ablation, diversity ablation
viewer/                   dock.js engine + docking_bench.html
slurm/                    array job that exercises checkpoint/resume
tools/                    viewer data export
tests/                    39 tests
```

## Licence

**Copyright (c) 2026 Arif Onur Alp. All rights reserved.** The source is public
so it can be read and evaluated; it is not licensed for reuse. You may read it,
run it locally to reproduce the results reported here, and quote short excerpts
with attribution. Anything beyond that needs permission, which is usually given
for academic use — see [LICENSE](LICENSE).

Bundled third-party components keep their own licences and are unaffected by
the above; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### A note on what a licence can and cannot do

A web app cannot be made uncopyable. The browser has to receive the JavaScript
in order to run it, so anyone who loads the deployed page already has the
source, and minifying it only costs a determined reader a few minutes. What a
licence does is make reuse *a decision someone has to defend*, rather than an
assumption. For work like this, public authorship with timestamps — a git
history, a dated deployment — is the practical protection against someone
passing it off as their own, and this notice is the legal one.
