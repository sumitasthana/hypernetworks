# UnCLe: reproduction plan and status

Working document for whoever picks this up next, human or agent. Reproduce
[An Unlearning Framework for Continual Learning](https://arxiv.org/abs/2509.17530)
(arXiv:2509.17530), and while doing it, collect the evidence our own hypothesis
needs. Several of the paper's own experiments double as controls for that
hypothesis; those are marked **dual purpose** and should not be run twice.

Last updated 2026-09-21 at commit `ca398ea`.

---

## 1. Our hypothesis

> Does UnCLe remove a forgotten task's information, or only remove access to
> it? And if it is recoverable, which components of the model hold it?

The second half is the sharper one. It turns a yes/no question into attribution.

What is *not* being claimed: that the hypernetwork trunk is untouched by
forgetting. It is not. Equation 3 and Algorithm 2 both optimise the
hypernetwork parameters, and `UnCLe.forget` does the same. Any framing that
rests on "only the task code changes" is wrong and must not reappear in a
write-up.

Candidate components, and whether forgetting touches them:

| Component | Size (ResNet50, 200 chunks) | Touched by `forget`? |
| --- | --- | --- |
| Task embedding | 32 numbers per task | No. Not passed to the optimiser |
| Shared trunk | 3 layers, 128/256/512 | Yes |
| Three output heads | most of 134,015,898 params | Yes |
| Chunk codes | 200 x 32 | No. Frozen after the first task |
| Per-task BatchNorm buffers | 53 pairs of running stats | **No, deliberately** |

Start with the buffers. That row is not a hypothesis: `uncle/trainer.py:forget`
leaves `task_buffers` alone on the grounds that eq. 3 covers generated
parameters and batch-norm statistics are buffers. Those statistics are computed
from the forgotten task's data and survive deletion intact.

Today's run narrowed the target usefully: the forget loss settled at its
theoretical floor of `gamma * d` = 235,208, so generated weights really do
collapse toward zero. The forgotten task's *output* therefore carries little,
and recovery, if any, must come from the components above.

Prior art to position against: relearning attacks on approximate unlearning are
established (Hu et al., ICLR 2025). The contribution must be the
task-conditioned hypernetwork setting and the component attribution, not the
general observation that relearning can reverse unlearning.

---

## 2. Status

### Works, verified

- The method: chunked generation, learn, forget, the shared regularizer,
  per-task BatchNorm statistics, the paper's Table 4 request sequences.
- Tiny ImageNet and Permuted MNIST task streams. One loader, one class
  partition (seed 42, `uncle/task_partition.json`, verified on every run).
- Per-request telemetry: seconds, seconds per step, peak GPU memory, protected
  count, written to disk as the run goes.
- Checkpoint after every request. A killed run resumes and reproduces the
  history it would have had, exactly, verified on CPU.
- 40 checks across `tests/`, plus 18 executable blocks in the Colab guide via
  `scripts/check_guide.py`.

### Broken, blocking everything

**Forgetting destroys the retained tasks.** One full run of sequence 1 on Tiny
ImageNet, ResNet50, A100, 56 minutes:

| Metric | Ours | Paper | |
| --- | --- | --- | --- |
| Retain accuracy | 10.00% | 55.24% | failed |
| Forget accuracy | 10.00% | 10.00% | vacuous, everything is at chance |
| Mean spill | 30.72 | 0.722 | 43x worse |
| Mean relapse | 0.00 | 2.233 | vacuous |

Learning itself works: individual tasks reached 50-60% before the collapse.
Spill per unlearn request was `15.4, 162.8, 59.4, 48.4, 13.0, 63.0, 2.0, 4.0,
0.6, 0, 0, 0`; the trailing zeros are because nothing is left to damage. By
request 15 every task sits at chance, and requests 25-29 are learns with final
loss 2.3026 = ln(10), meaning the generator can no longer learn anything.

Full record: `docs/run-log.html`.

### Not implemented

- CIFAR-100 and 5-Tasks datasets (no loaders).
- The three alternative noising strategies from Appendix E.
- Membership inference attack (Appendix G).
- A path to adapt or relearn a forgotten task. `hypernet.add_task` raises on a
  task that already has a code, and `Config` rejects learning a task twice.
- The seven baseline unlearning methods. See decision D1.

---

## 3. Decisions already made

Do not re-litigate these without a reason. Each was decided deliberately.

- **D1. Do not reimplement the seven baselines** (BadTeacher, SCRUB, SalUn,
  JiT, GKT, SSD, CLPU). Weeks of work to re-establish an undisputed result.
  Cite the paper's Tables 1 and 2. Revisit only if a reviewer demands it.
- **D2. Classifier output scale is `1/fan_in`, not He.** He is derived for a
  hidden layer feeding a ReLU; on a layer whose outputs are logits it makes
  them far too large. With He the `cnn` backbone could not learn at all. This
  is a deliberate departure from the paper's Hyperfan initialization, which
  would target the same distribution He does and would not fix it.
- **D3. Class partition is fixed** at seed 42 in `uncle/task_partition.json`.
  The paper never says which classes form each task. Changing it invalidates
  comparison with every number we have.
- **D4. Accuracy is validation accuracy.** Tiny ImageNet's public test split
  has no labels. Say so in any write-up.
- **D5. One Tiny ImageNet loader.** `uncle/data.py` routes to
  `uncle/streams.py`. There used to be two with different partitions.

---

## 4. Traps already paid for

Do not reintroduce these. Each cost real time.

1. `sys.path.insert` before a directory exists poisons the import cache. Call
   `importlib.invalidate_caches()` after creating or installing anything.
2. Output filenames must carry what a sweep varies. They carry sequence,
   backbone and seed. Sweeping beta or gamma with `output` set still collides,
   so use `output=None` for those and keep the returned numbers.
3. `run_experiment` returns a dict, not a tuple.
4. Do not put the dataset on a mounted drive. 120,000 small files stalls.
   Dataset local, results on Drive.
5. `TinyImageNet` must be given `classes=` or it indexes all 200.
6. `preserve` caches the frozen reference per request. If you change the
   snapshot or a protected task's code mid-request, clear `_reference`.
7. A request's `before` is the previous request's `after`, reused deliberately.
8. Writing guide code without executing it has failed every single time.
   `scripts/check_guide.py` exists for this.

---

## 5. Plan

Cost figures are A100-hours from the model in `docs/run-log.html`, fitted to the
measured run and accurate to 0.1% on it. Halve for H100.

### Phase 0. Unblock. Nothing downstream is meaningful until this passes.

**T0.1 Gamma probe.** *0.2 h.*
Three requests (`L3 L0 U3`), gamma in `(0.1, 0.01, 0.001, 0.0001)`, ResNet50,
sequence 1, `output=None`.
- Record for each: task 0's accuracy before and after the forget, and
  `mean_spill`.
- **Accept:** a gamma exists where task 3 lands at or below 12% and task 0
  moves by less than 5 points. Then it is tuning, and T0.3 is unnecessary.
- **If no gamma works:** it is structural. Go to T0.2.

**T0.2 Structural check on the forget objective.** *0.5 h.* Only if T0.1 fails.
- Compare `preserve` summed over protected tasks against averaged
  (`uncle/trainer.py` divides by `len(protected)`). Averaging weakens per-task
  protection as history grows, though it cannot explain damage at the first
  forget where there is one protected task.
- Compare burn-in 100 against 25 and 10 at fixed gamma. 100 Adam steps at 1e-3
  on a shared generator is a large move.
- Check whether the noise target should be the scaled parameters rather than
  the raw generator output. We align `raw`; eq. 3 is written over `H(e_f)`.
  This changes the magnitude of the objective by the scale factors.
- **Accept:** one change brings spill on a three-request run below 5.

**T0.3 Re-run the reference.** *0.93 h.*
Sequence 1, Tiny ImageNet, ResNet50, seed 0, with whatever T0.1 or T0.2
selected.
- **Accept:** retain accuracy above 40%, mean spill below 3, and no learn
  request with final loss within 0.01 of 2.3026.

### Phase 1. The paper's headline, on the dataset we have.

**T1.1 Sequence 1, three seeds, Tiny ImageNet.** *2.8 h.*
- **Accept:** retain accuracy within 10 points of 55.24, forget accuracy within
  1 of 10.00, mean spill below 2, and the seed-to-seed spread reported.
- This is the number everything else is compared against. Do not proceed past
  a run that fails it.

**T1.2 Sequences 2 and 3, three seeds.** *5.6 h.*
- **Accept:** the four metrics hold across all three sequences. A method that
  works on one request ordering has not been shown to work.

### Phase 2. Dual-purpose infrastructure.

Each item here is required by the paper *and* produces evidence for our
hypothesis. Build these before Phase 3, not after.

**T2.1 BatchNorm buffer policy flag.** *Code, 0.5 day. Runs 2.8 h.*
**Dual purpose.**
- Paper side: what happens to a forgotten task's running statistics is
  unspecified. The README already lists it as an open choice.
- Hypothesis side: those statistics derive from the forgotten task's data and
  currently survive deletion. This is the most likely leak.
- Add `forget_resets_buffers` to `Config`, default False to preserve current
  behaviour. When True, `forget` resets that task's buffers to
  `buffer_template`.
- **Accept:** sequence 1, three seeds, both settings. Report all four metrics
  for each. A difference in forget accuracy between them is direct evidence
  that buffers carry task information.

**T2.2 Alternative noising strategies (Appendix E).** *Code, 1 day. Runs 2.8 h.*
**Dual purpose.**
- Paper side: Table 8 compares fixed noise, norm reduction and discarding the
  embedding against UnCLe's averaged fresh draws.
- Hypothesis side: "discard `e_f`" replaces the forgotten task's embedding with
  a random one. If discarding prevents recovery where UnCLe's version does not,
  the embedding was carrying the information. That is a component result, free
  with a paper experiment.
- Implement as a `forget_objective` field: `"noise_average"` (current),
  `"noise_fixed"` (one draw sampled outside the loop, **not** the same as
  `noise_samples=1`, which draws fresh each step), `"norm_reduce"`,
  `"discard_embedding"`.
- **Accept:** reproduces Table 8's ordering on Tiny ImageNet, namely fixed
  noise loses retain accuracy, norm reduction barely forgets, UnCLe gets both.
  Note that `UNI` in that table is never defined in the paper and cannot be
  reproduced.

**T2.3 Adaptation path for a forgotten task.** *Code, 0.5 day.*
Required by our hypothesis, not by the paper.
- New entry point, not `learn`. Takes a task that already has a code, and an
  explicit embedding policy: `reuse`, `reset_random`, `freeze`.
- Must also allow freezing subsets of the generator, so embedding-only
  adaptation can be separated from generator adaptation.
- **Accept:** a unit test adapts a forgotten task on 50 examples and its
  accuracy rises, with the retained tasks unchanged to within a point.

### Phase 3. The rest of the paper's grid.

Runs only, no new code, on Tiny ImageNet unless a claim needs another dataset.

| Task | What | Cost |
| --- | --- | --- |
| T3.1 | Beta search, 4 values (Table 5) | 3.7 h |
| T3.2 | Gamma and burn-in search (Table 6) | 2.8 h |
| T3.3 | Burn-in annealing on and off (Table 7) | 1.9 h |
| T3.4 | ResNet18 variants, seq 1, 3 seeds (Appendix H) | 1.3 h |
| T3.5 | Saturation: the learning-only arm, 3 seeds (Table 3) | 2.5 h |

**T3.2 note:** copy the paper's selection rule exactly. First keep only the
gamma values driving forget accuracy to chance or below, *then* among those pick
the highest retain accuracy. Forgetting is the constraint; retention is what you
optimise subject to it. On Permuted MNIST the paper's own table has no value
satisfying "at or below 10", so decide and record your tie-break before running.

**T3.5 note:** score both arms on the tasks the *real* sequence still holds at
the end, not on all twenty. The learning-only arm never gives any up, so
comparing its average over twenty against the real run's over six is
meaningless. The paper's Figure 7 compares exactly the six survivors of
sequence 1.

### Phase 4. Hypothesis experiments.

Needs T2.3. Reference runs are the only expensive part; the pre-deletion
positive control is a checkpoint, so it is free.

**T4.1 Paired-history references.** *12.6 h.*
For each of 5 probed tasks and 3 seeds, run sequence 1 with that task's learn
and forget requests removed. These are the never-learned-X baselines.
- Record the exact request list used, since removing requests changes what
  every later request protects.

**T4.2 Recovery curves.** *3.8 h for the 2 adaptation modes A prescribes, 10 h for the 5 or 6 that attribution needs.*
Adapt three models on the same held-out examples of task X, with identical
budgets: the unlearned model, the paired reference, and the pre-deletion
checkpoint as a positive control. Several class-balanced budgets. Evaluate on a
disjoint X split.
- **Read:** faster recovery than the paired reference is evidence consistent
  with residual influence. No advantage means *this probe* found none, which is
  not proof of deletion. Write it that way.

**T4.3 Component attribution.** *Included in T4.2's larger figure.*
Repeat T4.2 with one component adaptable at a time: embedding only, trunk only,
heads only, and with buffers reset against preserved (uses T2.1).
- **Accept:** a ranking of components by recovery speed, with the buffers
  condition separated out.

**T4.4 Repeat after later learning.** *Free, reuses T4.1 runs.*
Probe immediately after deletion and again at the end of the sequence. Relapse
says a forgotten task's accuracy does not drift up on its own; it does not say
the task is not easier to relearn.

### Phase 5. Other datasets. Optional.

Only if a claim needs them. Each is code plus compute.

| Task | Cost | Note |
| --- | --- | --- |
| T5.1 CIFAR-100 loader and runs | 1 day + 1.0 h | Cheapest per run of the four |
| T5.2 5-Tasks loader and runs | 2 days + 2.8 h | Five source datasets |
| T5.3 Permuted MNIST at full scale | 5.1 h | Loader exists, never run at scale. Most expensive per run: 4,690 steps per learn |
| T5.4 Membership inference (Appendix G) | 1 day + small | Paper's own version cannot separate methods, because a forgotten head is randomised for every method. The useful version attacks the shared trunk, which is our hypothesis, not theirs |

---

## 6. Budget

| Block | A100 h | On H100 at 2x |
| --- | --- | --- |
| Phase 0, unblock | 1.7 | 0.9 |
| Phase 1, headline | 8.4 | 4.2 |
| Phase 2, dual purpose | 5.6 | 2.8 |
| Phase 3, the grid | 12.2 | 6.1 |
| Phase 4, hypothesis | 22.6 | 11.3 |
| **Reproduction plus hypothesis, Tiny ImageNet** | **50.4** | **25.2** |
| Phase 5, other datasets | 8.9 | 4.5 |

Roughly 4 days of engineering across Phases 2 and 3, and the hypothesis write-up
is the long pole, not the compute. **Compute is not the constraint at any point
in this plan.**

---

## 7. How to verify anything

```bash
python tests/test_uncle.py          # 14 checks, seconds
python tests/test_tasks.py          # 8 checks, seconds
python tests/test_experiments.py    # 18 checks, ~2 min, needs the dataset
python scripts/check_guide.py       # executes 18 guide code blocks, a few min
```

All four must pass before any commit. Every run writes four JSON files named for
sequence, backbone and seed: `history_`, `summary_`, `costs_`, `environment_`.
The environment file records the GPU, torch version and git commit, so a number
found later can be traced to the code that produced it.

Collapse signature, seen twice now from different causes: **final loss 2.3026
with accuracy exactly 10.0** means the network emits identical logits for every
input. The only question is what pushed it there.

---

## 8. Open questions for the team

- **Q1.** Do we need the seven baselines, or do we cite the paper? See D1.
  Affects the budget by roughly a factor of four and weeks of engineering.
- **Q2.** Do we claim anything on CIFAR-100 or 5-Tasks, or is Tiny ImageNet
  plus Permuted MNIST enough? Phase 5 hangs on this.
- **Q3.** For the hypothesis, is the deliverable an evaluation protocol or a
  claim about UnCLe specifically? The second needs the reproduction to hold
  first, which is why Phase 1 gates Phase 4.
- **Q4.** Does anyone want Hyperfan initialization implemented for fidelity?
  It would not fix D2's problem and it changes every number.
