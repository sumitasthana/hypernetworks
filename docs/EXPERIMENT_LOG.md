# UnCLe experiment log: learning and selective-forgetting diagnostics

This record separates user-reported GPU experiments from local software checks.
The GPU experiments were run in Colab; their complete artifacts remain on the
user's Drive or in earlier runtime-local directories. The tables below transcribe
results supplied in the conversation. They are not newly reproduced measurements.

Current status: **no tested forgetting trajectory has passed both the forgetting
and retention criteria. E10 is prepared but has no reported result.**

Start the next session with [03_forgetting_diagnostics.ipynb](../notebooks/03_forgetting_diagnostics.ipynb).
It restores the existing checkpoint and uses `uncle.diagnose_forgetting`.

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
| E10 | Planned: restore same model; gamma 0.00001; forgetting-only LR 0.00001; ten steps | Not reported | Not reported | Run in the fresh notebook, then review. |

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

## Next action: E10 only

Open the fresh notebook, mount Drive, and restore the checkpoint above. Use
forgetting LR 0.00001, gamma 0.00001, and ten continuous updates. Keep the original
training configuration unchanged. Record the full trajectory with the same
learning, forgetting, and retained-drift gates. The notebook saves a separate
screening summary alongside the raw diagnostic report.

If no step passes, inspect the trajectory before selecting another change. If a
step passes, treat it as a candidate, then validate it beyond this checkpoint and
on the full sequence. Do not interpret the current evidence as a completed paper
reproduction, and do not automatically proceed to recovery experiments.
