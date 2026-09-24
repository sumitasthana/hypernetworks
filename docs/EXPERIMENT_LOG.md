# UnCLe experiment log: learning and selective-forgetting diagnostics

This record separates user-reported GPU experiments from local software checks.
The GPU experiments were run in Colab; their complete artifacts remain on the
user's Drive or in earlier runtime-local directories. The tables below transcribe
results supplied in the conversation. They are not newly reproduced measurements.

Current status: **no tested forgetting trajectory has passed both criteria.
E14 first failed retention at step 36 and never reached target accuracy at
or below 12%. Pause scalar tuning and inspect the objective's parameter scaling.**

Start the next session with [03_forgetting_diagnostics.ipynb](../notebooks/03_forgetting_diagnostics.ipynb).
It restores the existing checkpoint and uses `uncle.diagnose_forgetting`.

## Reading this record

Last updated: 2026-09-24. This is the narrative record of all results supplied
in this conversation, not an archive of every original Colab output file.
[Structured observations](experiments/README.md) provide CSV traces for E08-E14,
sampled E13 losses, and a manifest linking the supplied artifact paths.
Earlier learning results and request summaries appear below; missing
measurements are not reconstructed.

Run identifiers are the mentoring sequence. Some early directory labels were
reused, so a label alone does not verify a configuration. Settings described
as intended or instructed must be checked against original report JSON before
using them as independently verified experimental metadata.

The latest Colab session expired. The model and diagnostic reports were saved
on Drive, but their current availability has not been checked from this machine.
In a fresh GPU runtime, run notebook sections 1-3 only to mount Drive and inspect
the checkpoint. Expected saved accuracies: task 3 = 26.0%, task 0 = 44.6%.
Stop before section 4, which contains a saved E14 run with older E10 explanatory text.
The proposed parameter-group diagnostic is not implemented yet.

## Question and protocol

Can the shared hypernetwork forget task 3 while preserving task 0?

The diagnostic sequence is `L3 L0 U3`: learn task 3, learn task 0, forget task 3.
Tiny ImageNet tasks use the saved seed-42 class partition, ten classes per task,
and task-local labels. Accuracy is validation accuracy, not labeled test accuracy.
Full task splits contain 5,000 training and 500 validation images.

The notebook configuration specifies ResNet50, 200 chunks, hidden widths
128/256/512, 32-dimensional task and chunk codes, five learning epochs, batch
size 64, evaluation batch size 256, beta 0.01, ten noise samples, and seed 0.
These are the intended common settings; the repository has not independently
inspected every earlier run's environment JSON. Where included in the pasted
logs, the environment was NVIDIA L4, 22.0 GiB, PyTorch 2.11.0+cu128, commit
`4ff99e8`. That commit predates the diagnostic-helper refactor.

Screening criteria, fixed before interpreting the short runs:

- Both task accuracies must be at least 25% immediately before forgetting.
- Task 3 must finish at or below 12%.
- Task 0's absolute accuracy change must be less than five percentage points.

The 25% threshold is a practical learning screen chosen during mentoring. It is
not a paper reproduction target or a statistical test. The 12% and five-point
criteria follow the short-probe plan. A pass would identify a candidate requiring
full-sequence validation, not prove information deletion.

## Run summary

Here, LR means learning rate. Before/after refers to the forget request unless
the row explicitly describes learning only. Each early run trained a fresh model;
starting accuracies varied despite the intended fixed seed. The cause of that
variation was not established.

| Run | Change and budget | Task 3 | Task 0 | Decision |
| --- | --- | --- | --- | --- |
| Initial probe | Shared LR 0.001; gamma 0.01; 100 forget steps | 14.4% after first learning; 12.2% before forgetting; 10.0% after | 36.6% to 34.8% | Invalid: target failed the learning screen. |
| Direct baseline | Train task 3 directly with ResNet50; five epochs, LR 0.001 | Validation by epoch: 38.4, 24.2, 55.6, 47.0, 53.8% | Not trained | Direct training can learn task 3 within this budget. |
| E02 | Hypernetwork, task 3 only; learning LR 0.0001 | 32.0%; final epoch loss 1.574293 | Not trained | Learning screen passed. |
| E03 | Learn tasks 3 and 0; learning LR 0.0001 | 32.0% after own learning, 30.4% after task 0 | 43.8% | Both tasks passed; no forget request. |
| E04 | Shared LR 0.0001; gamma 0.01; 100 steps | 31.0% after first learning; 26.4% to 10.0% at forgetting | 41.6% to 10.0% | Fail; spill 31.6 points. |
| E05 | Same budget; gamma 0.001 | 38.2% after first learning; 35.8% to 10.0% | 34.8% to 10.0% | Fail; spill 24.8 points. |
| E06 | Same budget; gamma 0.0001 | 42.6% after first learning; 35.2% to 10.0% | 37.0% to 10.0% | Fail; spill 27.0 points. |
| E07 | Gamma 0.0001; shared LR 0.0001; ten steps, floor set to ten | 37.0% after first learning; 39.6% to 10.0% | 48.8% to 10.0% | Fail; spill 38.8 points. Damage occurs within ten updates. |
| E08 | Saved pre-forgetting model; gamma 0.0001; forgetting LR 0.0001; ten measured updates | 26.0% to 10.0% | 44.6% to 10.0% | Fail; first update already damages task 0 by 21.2 points. |
| E09 | Restore E08 model; gamma 0.0001; forgetting-only LR 0.00001; ten measured updates | 26.0% to 15.8% | 44.6% to 30.0% | Fail; smaller updates delay damage, but no measured step passes both criteria. |
| E10 | Same saved model; gamma 0.00001; forgetting-only LR 0.00001; ten steps; Tesla T4 | 26.0% to 16.0% | 44.6% to 44.8% | Fail: retention passed, target remained above 12%. |
| E11 | Same checkpoint, LR 0.00001 and gamma 0.00001; 30 updates | 26.0% to 13.4%; minimum 12.6% | 44.6% to 36.4% | Fail: retention first breached at step 23; no target pass. |
| E12 | Same checkpoint, LR 0.00001 and 30-step budget; gamma 0.000003 | 26.0% to 13.0% | 44.6% to 42.8% | Fail: target above 12%; retention passed throughout. |
| E13 | Same checkpoint, LR 0.00001 and gamma 0.000003; 50 updates | 26.0% to 13.2%; minimum 13.0% | 44.6% to 43.8% | Fail: retention passes throughout; extra updates do not reach 12%. |
| E14 | Same checkpoint, intended LR 0.00001 and gamma 0.000005; 50 updates | 26.0% to 12.6%; minimum 12.4% | 44.6% to 37.4% | Fail: retention first breached at step 36; target never passed. |

E06 reused a directory whose label began `E04_first_forget`; the user clarified
that the hyperparameters changed while the label was reused. We record the gamma
as user-confirmed, not inferred from that directory name or the loss value.

Selected reported costs and losses:

| Run | Final forgetting loss | Forget time | Other observations |
| --- | --- | --- | --- |
| E04 | 565622.8750 | 22.6 s / 100 steps | Learning requests took about 95 and 140 s. |
| E05 | 53909.2578 | 22.6 s / 100 steps | Lower loss after changing gamma is not evidence of better forgetting. |
| E06 | 6864.8228 | 22.6 s / 100 steps | Both tasks still ended at chance. |
| E07 | 36058.9922 | 3.1 s / ten steps | Total run about four minutes; most time was repeated learning. |

These timings do not predict the new diagnostic's runtime: per-step evaluation
adds work. The short runs reported peak allocated GPU memory around 4.64 GiB.

## Persistent starting checkpoint

An earlier Colab session ended, losing its in-memory trainer and runtime-local
files. E04 through E07 had disabled checkpoint saving, so their JSON logs could
not restore their model states. The two learning requests were run again and
saved persistently:

```text
/content/drive/MyDrive/uncle/E08_forgetting_trace/before_forgetting.pt
```

This model reached 36.4% on task 3 after its first learn. After learning task 0,
the saved accuracies were task 3 = 26.0% and task 0 = 44.6%. Those are the starting
values for E08 and E09. Restored predictions were checked and matched both values.

E08 and E09 use one continuous Adam optimizer and one frozen reference within
each forget request. Evaluation preserves PyTorch RNG state and restores model
modes. Calling `forget(burn_in=1)` repeatedly would reset the optimizer and
reference, and would be a different experiment.

## E08: trace at forgetting LR 0.0001, gamma 0.0001

Loss components are measured **before** the update. Accuracies and drift are
measured **after** the update. Drift is task 0's absolute change from 44.6%.

| Step | Task 3 % | Task 0 % | Drift, points | Weighted noise before | Preservation before |
| --- | --- | --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 | N/A | N/A |
| 1 | 16.0 | 23.4 | 21.2 | 38765.68 | 0.00 |
| 2 | 13.0 | 14.6 | 30.0 | 36631.19 | 711.19 |
| 3 | 12.6 | 13.6 | 31.0 | 34988.89 | 1559.62 |
| 4 | 12.8 | 13.6 | 31.0 | 33794.32 | 2083.34 |
| 5 | 13.8 | 14.2 | 30.4 | 32940.63 | 2225.69 |
| 6 | 14.4 | 12.4 | 32.2 | 32308.67 | 2088.67 |
| 7 | 14.4 | 11.2 | 33.4 | 31802.68 | 1809.40 |
| 8 | 11.2 | 10.8 | 33.8 | 31353.69 | 1499.43 |
| 9 | 10.2 | 10.2 | 34.4 | 30910.52 | 1225.50 |
| 10 | 10.0 | 10.0 | 34.6 | 30435.68 | 1015.46 |

No step passed. Reported artifact, relative to the persistent checkpoint directory:
`forget_trace_20260923_020958_555174.json`.

## E09: trace at forgetting LR 0.00001, gamma 0.0001

| Step | Task 3 % | Task 0 % | Drift, points | Weighted noise before | Preservation before |
| --- | --- | --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 | N/A | N/A |
| 1 | 23.0 | 45.0 | 0.4 | 38765.68 | 0.00 |
| 2 | 21.2 | 46.6 | 2.0 | 38546.75 | 7.25 |
| 3 | 18.8 | 44.8 | 0.2 | 38331.81 | 27.22 |
| 4 | 17.4 | 43.4 | 1.2 | 38116.39 | 57.94 |
| 5 | 16.0 | 41.8 | 2.8 | 37906.20 | 97.46 |
| 6 | 16.2 | 39.2 | 5.4 | 37700.46 | 144.03 |
| 7 | 16.6 | 37.0 | 7.6 | 37498.27 | 196.03 |
| 8 | 16.8 | 35.2 | 9.4 | 37302.28 | 252.03 |
| 9 | 16.4 | 32.6 | 12.0 | 37112.92 | 310.69 |
| 10 | 15.8 | 30.0 | 14.6 | 36926.21 | 370.77 |

No step passed. Reported artifact:
`E09_forget_lr_1e-5_20260923_022053_354220.json`.
The filenames carry UTC timestamps; this document preserves the supplied names.

## E10: retention holds over ten steps, target remains above threshold

User-reported on 2026-09-23, using Tesla T4. Settings are those prepared in
the E10 notebook: forgetting LR 0.00001, gamma 0.00001, ten steps. The supplied
screening table does not include the raw report settings or loss components;
those have not been independently inspected.

| Step | Task 3 % | Task 0 % | Absolute task 0 drift, points |
| --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 |
| 1 | 23.0 | 45.0 | 0.4 |
| 2 | 21.6 | 46.6 | 2.0 |
| 3 | 19.4 | 46.0 | 1.4 |
| 4 | 18.4 | 44.8 | 0.2 |
| 5 | 17.4 | 44.8 | 0.2 |
| 6 | 16.6 | 44.6 | 0.0 |
| 7 | 16.0 | 45.0 | 0.4 |
| 8 | 16.2 | 44.6 | 0.0 |
| 9 | 16.0 | 44.8 | 0.2 |
| 10 | 16.0 | 44.8 | 0.2 |

Passing steps: none. Task 0 stayed within the five-point drift limit at every
measured step. Task 3 fell ten points but remained four points above the target
threshold. The last four observations suggest a short plateau; ten updates do
not establish whether it persists.

Screening artifact supplied by the user:
`/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/forget_20260923_142742_335991_88cd96f2.screen.json`.

Compared with E09, the retained-task endpoint is better (44.8% versus 30.0%).
This is consistent with improved preservation at lower gamma, but the GPU also
changed from the earlier L4 environment to T4 and the diagnostic implementation
changed. Equal starting accuracies do not establish identical numerical update
trajectories. Do not attribute the whole difference to gamma without a matched
control. For the next budget comparison, keep this T4 runtime and helper fixed.

## E11: longer run exposes the retention tradeoff

User-reported on 2026-09-23, following the E11 instructions. The supplied table
covers 30 updates. LR 0.00001 and gamma 0.00001 are the instructed settings;
the raw configuration JSON has not been inspected. E11 was requested in the
same T4 runtime as E10; the user did not supply a separate GPU line for E11.

| Step | Task 3 % | Task 0 % | Absolute task 0 drift, points |
| --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 |
| 1 | 23.0 | 45.0 | 0.4 |
| 2 | 21.6 | 46.6 | 2.0 |
| 3 | 19.4 | 46.0 | 1.4 |
| 4 | 18.4 | 44.8 | 0.2 |
| 5 | 17.4 | 44.8 | 0.2 |
| 6 | 16.6 | 44.6 | 0.0 |
| 7 | 16.0 | 45.0 | 0.4 |
| 8 | 16.2 | 44.6 | 0.0 |
| 9 | 16.0 | 44.8 | 0.2 |
| 10 | 16.0 | 44.8 | 0.2 |
| 11 | 16.4 | 44.6 | 0.0 |
| 12 | 16.8 | 44.2 | 0.4 |
| 13 | 16.4 | 43.6 | 1.0 |
| 14 | 16.4 | 43.4 | 1.2 |
| 15 | 16.2 | 43.2 | 1.4 |
| 16 | 15.8 | 42.6 | 2.0 |
| 17 | 15.0 | 42.0 | 2.6 |
| 18 | 15.0 | 41.4 | 3.2 |
| 19 | 14.2 | 41.6 | 3.0 |
| 20 | 13.4 | 40.6 | 4.0 |
| 21 | 13.6 | 40.2 | 4.4 |
| 22 | 13.4 | 39.8 | 4.8 |
| 23 | 12.8 | 39.4 | 5.2 |
| 24 | 12.6 | 38.6 | 6.0 |
| 25 | 12.8 | 38.6 | 6.0 |
| 26 | 12.8 | 38.6 | 6.0 |
| 27 | 13.2 | 38.2 | 6.4 |
| 28 | 13.8 | 37.8 | 6.8 |
| 29 | 13.6 | 37.6 | 7.0 |
| 30 | 13.4 | 36.4 | 8.2 |

All first-ten-step accuracies match E10 exactly at the reported precision.
The apparent plateau near 16% was temporary. At step 22, task 3 was 13.4%
and task 0 was 39.8%, still within the retention limit. At step 23, retained
drift reached 5.2 points, failing the screen. The lowest target accuracy was
12.6% at step 24, with six points of retained drift. No target observation was
at or below 12%, and no step passed both criteria. By step 30 retained drift
was 8.2 points. Extending this run did not solve selective forgetting.

Reported screening artifact:
`/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/forget_20260923_143249_572038_f52e5457.screen.json`.

Code inspection confirms the forgetting loss is gamma times the noise loss
plus the preservation loss. Beta scales preservation during learning only;
changing beta would not change this forgetting objective. E12 therefore tests
a lower gamma, without changing the learning configuration. This may improve
retention, but may also leave more target accuracy. It is a hypothesis, not an
expected pass. Do not relax the screen because E11 came close.

## E12: retention passes throughout, target ends one point above threshold

User-reported on 2026-09-23 after the E12 instructions. The instructed settings
were forgetting LR 0.00001, gamma 0.000003, and 30 updates. Raw configuration
and loss components were not supplied for inspection. The same T4 runtime was
requested; no separate GPU identification was included in this result.

| Step | Task 3 % | Task 0 % | Absolute task 0 drift, points |
| --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 |
| 1 | 23.0 | 45.0 | 0.4 |
| 2 | 22.6 | 45.2 | 0.6 |
| 3 | 21.8 | 45.4 | 0.8 |
| 4 | 21.0 | 45.8 | 1.2 |
| 5 | 19.2 | 45.4 | 0.8 |
| 6 | 18.6 | 45.8 | 1.2 |
| 7 | 17.4 | 46.2 | 1.6 |
| 8 | 16.4 | 46.0 | 1.4 |
| 9 | 15.8 | 44.6 | 0.0 |
| 10 | 16.0 | 44.4 | 0.2 |
| 11 | 16.2 | 44.2 | 0.4 |
| 12 | 16.0 | 44.6 | 0.0 |
| 13 | 16.2 | 44.4 | 0.2 |
| 14 | 16.8 | 44.6 | 0.0 |
| 15 | 16.8 | 44.6 | 0.0 |
| 16 | 16.4 | 44.8 | 0.2 |
| 17 | 16.2 | 44.2 | 0.4 |
| 18 | 16.2 | 44.2 | 0.4 |
| 19 | 15.8 | 43.8 | 0.8 |
| 20 | 15.4 | 43.8 | 0.8 |
| 21 | 15.2 | 43.4 | 1.2 |
| 22 | 15.0 | 43.4 | 1.2 |
| 23 | 14.8 | 43.4 | 1.2 |
| 24 | 14.2 | 43.4 | 1.2 |
| 25 | 14.0 | 43.4 | 1.2 |
| 26 | 14.4 | 43.2 | 1.4 |
| 27 | 14.0 | 43.2 | 1.4 |
| 28 | 13.6 | 42.8 | 1.8 |
| 29 | 13.0 | 42.8 | 1.8 |
| 30 | 13.0 | 42.8 | 1.8 |

No step passed both criteria. All steps passed retention; target accuracy
never reached 12%. Task 3 ended at 13.0%, one percentage point above the target,
while task 0 drift was 1.8 points. Compared with E11 at step 30, the endpoints
were 13.0% versus 13.4% for task 3 and 42.8% versus 36.4% for task 0. This is
an improved retention result on this checkpoint, not a validated optimum.

Task 3 declined from 15.4% at step 20 to 13.0% at step 30. That trend and the
remaining retention margin motivate a bounded extension before changing gamma
again. They do not guarantee continued improvement or a passing step. The
loss components remain uninspected; no gradient-dominance claim follows.

Reported screening artifact:
`/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/forget_20260923_143653_273412_d31bb28c.screen.json`.

## E13: retention holds, but extra steps do not reach the target

User-reported on 2026-09-23 after the E13 instructions. Intended settings were
forgetting LR 0.00001, gamma 0.000003 and 50 updates. Raw settings and losses
have not yet been inspected. The same T4 runtime was requested; no separate
GPU identification was supplied with this result.

| Step | Task 3 % | Task 0 % | Absolute task 0 drift, points |
| --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 |
| 1 | 23.0 | 45.0 | 0.4 |
| 2 | 22.6 | 45.2 | 0.6 |
| 3 | 21.8 | 45.4 | 0.8 |
| 4 | 21.0 | 45.8 | 1.2 |
| 5 | 19.2 | 45.4 | 0.8 |
| 6 | 18.6 | 45.8 | 1.2 |
| 7 | 17.4 | 46.2 | 1.6 |
| 8 | 16.4 | 46.0 | 1.4 |
| 9 | 15.8 | 44.6 | 0.0 |
| 10 | 16.0 | 44.4 | 0.2 |
| 11 | 16.2 | 44.2 | 0.4 |
| 12 | 16.0 | 44.6 | 0.0 |
| 13 | 16.2 | 44.4 | 0.2 |
| 14 | 16.8 | 44.6 | 0.0 |
| 15 | 16.8 | 44.6 | 0.0 |
| 16 | 16.4 | 44.8 | 0.2 |
| 17 | 16.2 | 44.2 | 0.4 |
| 18 | 16.2 | 44.2 | 0.4 |
| 19 | 15.8 | 43.8 | 0.8 |
| 20 | 15.4 | 43.8 | 0.8 |
| 21 | 15.2 | 43.4 | 1.2 |
| 22 | 15.0 | 43.4 | 1.2 |
| 23 | 14.8 | 43.4 | 1.2 |
| 24 | 14.2 | 43.4 | 1.2 |
| 25 | 14.0 | 43.4 | 1.2 |
| 26 | 14.4 | 43.2 | 1.4 |
| 27 | 14.0 | 43.2 | 1.4 |
| 28 | 13.6 | 42.8 | 1.8 |
| 29 | 13.0 | 42.8 | 1.8 |
| 30 | 13.0 | 42.8 | 1.8 |
| 31 | 13.0 | 42.8 | 1.8 |
| 32 | 13.2 | 42.8 | 1.8 |
| 33 | 13.2 | 42.8 | 1.8 |
| 34 | 13.4 | 42.8 | 1.8 |
| 35 | 13.6 | 42.8 | 1.8 |
| 36 | 13.8 | 42.8 | 1.8 |
| 37 | 13.8 | 42.8 | 1.8 |
| 38 | 14.0 | 42.8 | 1.8 |
| 39 | 13.8 | 42.8 | 1.8 |
| 40 | 13.2 | 43.0 | 1.6 |
| 41 | 13.4 | 43.0 | 1.6 |
| 42 | 13.8 | 43.0 | 1.6 |
| 43 | 13.0 | 43.2 | 1.4 |
| 44 | 13.4 | 43.2 | 1.4 |
| 45 | 13.4 | 43.4 | 1.2 |
| 46 | 13.4 | 43.4 | 1.2 |
| 47 | 13.4 | 43.4 | 1.2 |
| 48 | 13.2 | 43.4 | 1.2 |
| 49 | 13.0 | 43.4 | 1.2 |
| 50 | 13.2 | 43.8 | 0.8 |

The first 30 accuracy pairs match E12 at the reported precision. After step
30, task 3 stays between 13.0% and 14.0%. Its minimum across the run is 13.0%,
so no step reaches the 12% threshold. Task 0 meets retention at every step
and finishes only 0.8 points below its starting accuracy. The 20 extra updates
did not improve the lowest observed target accuracy.

This is an accuracy plateau over the measured window, not evidence that the
loss or parameters converged. The two tasks' accuracies need not change
monotonically. Keep the existing thresholds; do not redefine a pass because
the target came close. No claim about erasure follows from these measurements.

Reported screening artifact:
`/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/forget_20260923_143926_526429_aa83a19b.screen.json`.

## E13 loss follow-up: objective decreases despite the accuracy plateau

User supplied these sampled values after reading E13's report. All losses
are measured before the corresponding update. The requested settings and
status output were not included; exact run settings remain based on the
experiment instructions rather than an independently inspected configuration.

| Step | Weighted noise | Preservation | Total loss |
| --- | --- | --- | --- |
| 1 | 1163.01 | 0.00 | 1163.01 |
| 10 | 1140.43 | 3.66 | 1144.09 |
| 20 | 1117.63 | 4.15 | 1121.78 |
| 30 | 1095.24 | 5.29 | 1100.53 |
| 40 | 1073.38 | 6.50 | 1079.88 |
| 50 | 1052.21 | 7.66 | 1059.87 |

The sampled total falls by 103.14, about 8.9%, from step 1 to step 50.
Weighted noise decreases while preservation increases. This is consistent
with reducing the noise objective while generated protected weights move
away from their reference. It does not establish optimizer convergence,
relative gradient strength, or the accuracy effect of further updates.
Fresh noise samples also contribute to differences between loss observations.
The objective acts on generated weights, not directly on validation accuracy;
therefore continued loss reduction need not cross the accuracy screen.

Recommendation: test gamma 0.000005 with the E13 learning rate and 50-step
budget. This is between 0.000003, which preserved task 0 but missed target
accuracy, and 0.00001, which damaged retention in E11's 30-step run. These
outcomes motivate an intermediate test but do not guarantee monotonic behavior
or a feasible gamma. No thresholds are changed.

## E14: intermediate gamma still misses the joint screen

User-reported after the E14 instructions: intended gamma 0.000005, forgetting
LR 0.00001, and 50 updates from the same pre-forgetting checkpoint. The supplied
screen does not independently confirm the configuration or GPU.

| Step | Task 3 % | Task 0 % | Absolute task 0 drift, points |
| --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 |
| 1 | 23.0 | 45.0 | 0.4 |
| 2 | 22.0 | 45.6 | 1.0 |
| 3 | 21.0 | 46.0 | 1.4 |
| 4 | 19.4 | 46.2 | 1.6 |
| 5 | 18.6 | 46.4 | 1.8 |
| 6 | 18.0 | 46.6 | 2.0 |
| 7 | 17.0 | 46.2 | 1.6 |
| 8 | 16.2 | 45.8 | 1.2 |
| 9 | 16.0 | 44.6 | 0.0 |
| 10 | 16.2 | 44.6 | 0.0 |
| 11 | 16.2 | 44.8 | 0.2 |
| 12 | 16.4 | 44.6 | 0.0 |
| 13 | 17.0 | 44.6 | 0.0 |
| 14 | 16.4 | 44.6 | 0.0 |
| 15 | 16.4 | 44.4 | 0.2 |
| 16 | 16.4 | 43.8 | 0.8 |
| 17 | 15.6 | 43.4 | 1.2 |
| 18 | 15.4 | 43.2 | 1.4 |
| 19 | 15.0 | 42.4 | 2.2 |
| 20 | 14.8 | 42.8 | 1.8 |
| 21 | 14.2 | 42.8 | 1.8 |
| 22 | 13.8 | 42.4 | 2.2 |
| 23 | 13.6 | 41.8 | 2.8 |
| 24 | 13.6 | 41.4 | 3.2 |
| 25 | 13.0 | 41.2 | 3.4 |
| 26 | 13.0 | 41.4 | 3.2 |
| 27 | 12.6 | 41.2 | 3.4 |
| 28 | 12.8 | 40.8 | 3.8 |
| 29 | 13.2 | 39.8 | 4.8 |
| 30 | 13.4 | 40.2 | 4.4 |
| 31 | 14.0 | 40.2 | 4.4 |
| 32 | 14.0 | 40.0 | 4.6 |
| 33 | 13.8 | 40.0 | 4.6 |
| 34 | 13.6 | 39.8 | 4.8 |
| 35 | 13.0 | 39.8 | 4.8 |
| 36 | 13.2 | 39.2 | 5.4 |
| 37 | 13.4 | 39.0 | 5.6 |
| 38 | 13.0 | 39.2 | 5.4 |
| 39 | 13.0 | 39.0 | 5.6 |
| 40 | 12.8 | 38.6 | 6.0 |
| 41 | 13.0 | 38.0 | 6.6 |
| 42 | 13.0 | 38.0 | 6.6 |
| 43 | 12.6 | 38.0 | 6.6 |
| 44 | 12.4 | 38.0 | 6.6 |
| 45 | 12.6 | 38.2 | 6.4 |
| 46 | 12.6 | 38.0 | 6.6 |
| 47 | 12.6 | 38.0 | 6.6 |
| 48 | 12.6 | 37.8 | 6.8 |
| 49 | 12.6 | 37.8 | 6.8 |
| 50 | 12.6 | 37.4 | 7.2 |

At step 27, task 3 reached 12.6% while task 0 retained 41.2% accuracy
(3.4 points of drift). Retention first failed at step 36, with 5.4 points of
drift. The lowest task 3 accuracy was 12.4% at step 44, when retention drift
was already 6.6 points. Final values were 12.6% and 37.4%, with 7.2 points of
retained drift. No step passed. Keep the 12% threshold unchanged.

Reported screening artifact:
`/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/forget_20260923_151049_593433_8a010c9b.screen.json`.

The measured gamma values expose a retention/forgetting tradeoff on this
checkpoint. This finite search does not prove that no useful gamma exists or
that a structural defect is the cause. It does justify switching from further
nearby scalar guesses to the planned objective audit.

### Initial code audit: the two terms use different parameter scales

`UnCLe.forget` uses `hypernet.raw_for(task)` for its noise term, before layer
scaling. `UnCLe.preserve` compares `weights_from_code`, after layer scaling.
`HyperNetwork` assigns scale 0.01 to one-dimensional parameters, 1/fan_in to
classifier matrices, and sqrt(2/fan_in) to other matrices.

For a protected parameter with raw change delta and scale s, its contribution
to preservation is s squared times delta squared. Thus a global gamma cannot
remove all differences in relative weighting across layers. This is a code
property, not evidence by itself that these scales caused E14's failure.
Neither loss magnitude nor this observation establishes gradient dominance.
Changing to a scaled noise objective would change the experiment and require
an explicit choice of noise scale; it must not be silently treated as a fix.

## What we learned, and what remains uncertain

1. **Initial learning was a prerequisite failure.** The original target accuracy
   was weak before any deletion. Its subsequent chance accuracy could not
   demonstrate successful removal of a learned ability.
2. **The direct classifier learned task 3.** Its 53.8% final accuracy directs
   investigation toward the hypernetwork training path and optimization. It
   does not identify a particular faulty component because initialization and
   parameterization also differ.
3. **A lower training LR helped in the reported runs.** Moving from 0.001 to
   0.0001 enabled the target to pass our learning screen. This is one-seed
   diagnostic evidence, not a robust hyperparameter optimum.
4. **Chance accuracy on both tasks is a failure of selective forgetting.** A
   smaller spill across fresh runs can simply reflect a lower starting accuracy.
   Likewise, smaller losses after changing gamma have different scales.
5. **The damaging motion starts immediately.** At the initial snapshot, squared
   weight-difference preservation has zero loss and zero gradient. It responds
   after drift occurs; the first E08 update already reduced retained accuracy by
   21.2 points. This is a property of this penalty, not proof of an implementation
   defect by itself.
6. **Smaller forgetting updates delay, but have not prevented, the tradeoff.** E09
   keeps task 0 within the drift limit through step 5, while task 3 remains above
   the forgetting threshold. Continuing to step 10 damages retention.
7. **Loss magnitudes do not establish gradient dominance.** Gradient measurements
   would be needed for that claim. Low weight drift also does not directly
   guarantee low accuracy drift.
8. **Chance accuracy does not prove erasure.** No recovery experiment has yet
   established whether residual task information remains.
9. **Zero relapse here is not a stability result.** These traces have no later
   learning after deletion.
10. **Preserve the trained starting state.** Checkpoint comparisons avoid spending
    roughly four minutes relearning the same pair and reduce confounding from
    variation between fresh training attempts.

## Software changes supporting the next session

- `Config.forgetting_learning_rate` separates forgetting from learning. Its
  default, `None`, retains the previous shared-rate behavior.
- `UnCLe.forget(..., on_step=...)` reports step zero and each update, isolating
  callback PyTorch RNG consumption and restoring model modes.
- `diagnose_forgetting(...)` restores a checkpoint, calls the existing forgetting
  loop, evaluates each step, and saves a unique report. It never retrains or
  overwrites the source checkpoint.
- Checkpoint loading now keeps tensors on CPU until restoration. CUDA RNG states
  are validated and supplied as CPU byte tensors; entries are never silently
  skipped. This addresses the user's GPU RNG restoration error. Invalid GPU
  counts or states produce explicit errors.
- Older checkpoints without `forgetting_learning_rate` remain readable.

Local verification of this refactor: 15 core checks, eight task checks, 18
experiment checks, and eight diagnostic checks passed. The additional CUDA RNG
regression was skipped because the local machine has no CUDA device. All 19
executable Colab-guide examples passed after supplying an isolated local IPython
dependency. These checks establish software behavior on the tested environment;
they do not reproduce the GPU research outcomes above.

### Evidence added from the saved notebook on 2026-09-24

The newer notebook commit `4407978` contains saved outputs that corroborate
E13 settings (LR 0.00001, gamma 0.000003, 50 steps, ten noise samples) and E14
settings (LR 0.00001, gamma 0.000005, 50 steps, ten noise samples). This updates
the earlier notes saying those settings had not been supplied. The E13 cell
reads its older saved report; it is not another E14 result.

Setup outputs report Tesla T4 and source commit `04c31b3`. E14's 50 printed
accuracy pairs match the conversation trace. All 50 weighted-noise and
preservation pairs are now in the CSV. Weighted noise falls from 1938.36 to
1731.68 while preservation rises from zero to 13.70. These sampled changes
do not establish relative gradient strength or explain the failure by themselves.
The original report JSON remains on Drive and was not read from this machine.

[Archived notebook outputs](experiments/notebook_saved_outputs.txt) retain this
evidence with its source commit and cell indices. The notebook has historical
E10 prose alongside E14 code and outputs; follow the current status here.

## Next priority: objective diagnostics (T0.2)

Pause further gamma and step sweeps. Preserve the current checkpoint and all
reports; do not retrain or relax the acceptance thresholds. The next useful
diagnostic should measure target and retained generated-weight changes by
parameter group, including their raw and scaled norms, and relate those to
accuracy. If measuring gradient contributions, distinguish them from losses.

These measurements should be implemented in the repository with a short
notebook call, preserving optimizer continuity and observational RNG behavior.
No new objective, instrumentation, or GPU run has been executed in this update.
The audit above is a starting hypothesis to test, not a confirmed root cause.
