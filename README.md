# Component-level Attribution of Residual Knowledge after Task Unlearning

Working title. An independent reproduction of **UnCLe**, the method introduced
in *An Unlearning Framework for Continual Learning* (Adhikari, Kumaravelu and
Srijith, 2025, [arXiv:2509.17530](https://arxiv.org/abs/2509.17530)), and a
study of which parts of the model still hold a task after that task has been
unlearned.

A note on names, since the two get conflated. *An Unlearning Framework for
Continual Learning* is the paper. **UnCLe** is the method it introduces, and
what this repository implements. There is no paper called UnCLe.

One hypernetwork produces the weights of a target network from a short code,
one code per task. Learning a task trains the hypernetwork to classify it.
Forgetting a task trains the hypernetwork to turn that task code into noise,
which needs no data at all.

## Where things stand

The implementation runs the paper's full setting end to end and **does not yet
reproduce its numbers**. One pass of sequence 1 on Tiny ImageNet returns 10%
retain accuracy against the paper's 55.24%: learning works, forgetting works,
and forgetting destroys the tasks it is supposed to preserve. That is a fault
in this code, not a finding about the method, and nothing else is worth
measuring until it is fixed.

[docs/PLAN.md](docs/PLAN.md) has the rest: what is done, what is next, what
each remaining experiment costs, and the decisions not to re-litigate.

The longer write-ups, the run log and the Colab walkthrough, are kept outside
the repository. Only the plan travels with the code.

## Run it

The defaults are the paper's Permuted-MNIST setting: ResNet18 generated in 200
chunks, 10 tasks, and request sequence 1 from Table 4. That wants a GPU.

```bash
pip install -e .                   # or: pip install -r requirements.txt
python main.py                     # the paper's setting
python main.py --sequence 2        # sequences 1, 2 and 3 are all from Table 4
```

For a quick check without a GPU, swap in the small stand-in network:

```bash
python main.py --backbone cnn --chunks 32 --epochs 1
python tests/test_uncle.py         # fourteen checks, seconds, no download
python tests/test_tasks.py         # eight checks on the partition and the guide
python tests/test_experiments.py   # eighteen checks, a couple of minutes, needs the images
python scripts/check_guide.py      # runs every code block in the Colab guide
```

When a run collapses to 10% on every task, start here. It varies gamma over the
first three requests and reports whether the damage is tuning or structural:

```bash
python scripts/gamma_probe.py --check   # validate the setup, train nothing
python scripts/gamma_probe.py           # about ten minutes on an A100
```

Tiny ImageNet downloads itself on first use (about 240 MB) and wants a GPU. It
runs the paper's 30-request sequences over 20 tasks, with beta 0.01:

```bash
python main.py --dataset tiny_imagenet --backbone resnet50
python main.py --dataset tiny_imagenet --sequence 3
```

The paper reports 96.87% retain accuracy and 10.00% forget accuracy for the
default setting, so those are the numbers to check against.

## What is where

| Path | What it holds |
| --- | --- |
| `uncle/config.py` | Every knob, plus validation of the request list |
| `uncle/data.py` | Permuted MNIST, and the route to Tiny ImageNet |
| `uncle/tinyimagenet.py` | The Tiny ImageNet reader, which never moves a file |
| `uncle/tasks.py` | The saved 20 x 10 class partition and task-local labels |
| `uncle/streams.py` | Those tasks, in the shape the trainer wants |
| `uncle/hypernet.py` | The target network and the network that generates its weights |
| `uncle/trainer.py` | `learn`, `forget`, and the regularizer they share |
| `uncle/metrics.py` | Retain accuracy, forget accuracy, spill, relapse |
| `uncle/experiment.py` | Works through a request sequence, returns records |
| `uncle/experiments.py` | The callable front door: one run, or a sweep of them |
| `uncle/telemetry.py` | What a run cost: time, peak GPU memory, sizes |
| `uncle/baseline.py` | One task, ordinary backprop, no hypernetwork |
| `main.py` | Command line for Permuted MNIST and Tiny ImageNet |
| `scripts/` | `run.py`, `baseline.py`, and the dataset exploration scripts |
| `notebooks/` | Dataset and task exploration, plus the original Colab notebook |
| `docs/PLAN.md` | Reproduction plan and status |
| `reference/uncle_minimal.py` | The same method in one flat file, for reading |
| `tests/` | `test_uncle.py`, `test_tasks.py`, `test_experiments.py` |

`reference/uncle_minimal.py` is not imported by anything. It exists so the method can be
read top to bottom in one sitting before meeting the package.

## Running it from Python

`main.py` is the older command line and still works. The callable front door,
which is what Colab wants, is this:

```python
from uncle.experiments import run_experiment, run_sequences

result = run_experiment(sequence=1, backbone="resnet50", epochs=5)
print(result["numbers"])     # retain, forget, spill, relapse
print(result["totals"])      # seconds per action, peak GPU memory
```

Any `Config` field passes through as a keyword argument, so a sweep is a loop
over calls. All three of the paper's sequences over three seeds, with a
progress bar and a comparison table at the end:

```python
results = run_sequences(sequences=(1, 2, 3), seeds=(0, 1, 2),
                        backbone="resnet50", epochs=5, output="results")
```

Or from the command line:

```bash
python scripts/run.py --sequence 1 2 3 --seed 0 1 2 --backbone resnet50
```

Each run writes four JSON files named after the sequence, backbone and seed:
the history, the four numbers, the per-request costs, and the environment it
ran in, down to the git commit. The history and the costs are rewritten after
every request, so a long run can be watched and survives a crash.

It also writes a checkpoint after every request, holding the hypernetwork, the
per-task batch-norm statistics, the loop's bookkeeping and the random number
generator's state. Start the same run again and it continues from the last
finished request, making the same draws it would have made had it never
stopped. Pass `checkpoint=False` to skip it, or `resume=False` to start over.

The Colab walkthrough, kept outside the repository, has the setup cells,
worked examples, and every experiment in the paper with what to look for.

## Diagnose forgetting from a checkpoint

Start with [the fresh diagnostic notebook](notebooks/03_forgetting_diagnostics.ipynb)
([open in Colab](https://colab.research.google.com/github/sumitasthana/hypernetworks/blob/main/notebooks/03_forgetting_diagnostics.ipynb)).
The [experiment log](docs/EXPERIMENT_LOG.md) records the results, limitations, and
findings through E14. No measured step passes both forgetting and retention
criteria. [Structured observations](docs/experiments/README.md) preserve the
supplied traces and their provenance. The next priority is objective diagnostics.
After a session restart, run notebook sections 1-3 only; section 4 still contains
the saved E14 run with older E10 prose, not the proposed parameter-group diagnostic.

Use a checkpoint saved after learning the target task and before forgetting it.
This restores the model, task buffers, and random state on every call. It runs
only forgetting, through the same `UnCLe.forget` method used by experiments.

```python
from uncle import diagnose_forgetting

# Restore the same trained model for every comparison. No learning is repeated.
# Change forgetting_lr or gamma here, rather than editing the training loop.
report = diagnose_forgetting(
    checkpoint=CHECKPOINT,
    task="3",
    root=DATA,               # extracted Tiny ImageNet directory
    forgetting_lr=1e-5,      # affects forgetting only
    gamma=1e-5,              # noise-loss coefficient
    steps=10,               # exactly ten continuous Adam updates
)
print(report["report_path"])
```

The helper prints step-zero accuracy and accuracy after every update. Noise
and preservation losses describe the model before that update. Evaluation
preserves the update's random-number stream and model modes. It does not reset
Adam or the frozen reference between steps. All other seen tasks are protected,
including previously forgotten tasks.

A uniquely named JSON report is saved under `diagnostics/` beside the checkpoint,
or in the directory passed as `output`. It records both training and diagnostic
settings, evaluation sizes, initial agreement with saved accuracies, and each
step's measurements. The source checkpoint is never overwritten. An interrupted
trace retains completed observations; a new call starts again from the source
checkpoint. Tiny ImageNet evaluation uses full validation splits, so initial
accuracy can differ from a checkpoint evaluated on capped images.

For ordinary experiment runs, pass `forgetting_learning_rate=1e-5` to
`run_experiment`. Its default is `None`, which preserves the previous behavior
of sharing `learning_rate` between learning and forgetting. Old checkpoints
without this field remain readable. Diagnostics can override the forgetting
rate without changing the saved training configuration.

Check the diagnostic machinery without downloading data:

```bash
python tests/test_diagnostics.py
```

## What a run costs

`uncle/telemetry.py` times each request and records peak GPU memory around it,
because learning and forgetting are very different jobs and averaging them
hides that. A finished run prints a table of both, per request, plus totals.

Peak memory is the number to watch on a long sequence. The regularizer
regenerates every protected task's weights inside one graph, so a request near
the end of a 30-request sequence is holding many times what the first one did.

## The four numbers

Accuracy alone hides what goes wrong when unlearning happens inside continual
learning, so the paper reports four.

- **RA**, retain accuracy. Average over tasks still held at the end. Higher is better.
- **FA**, forget accuracy. Average over forgotten tasks. Should sit at chance, 10%.
- **Spill**, how much a forget request disturbed every *other* task. Lower is better.
- **Relapse**, how much a forgotten task crept back as later tasks were learned. Lower is better.

Spill and relapse are the two failure modes the paper identifies in existing
unlearning methods. Neither shows up in accuracy at the end of a run.

## What the noise objective actually does

Equation 3 asks the hypernetwork to make a forgotten task's generated weights
match a Gaussian noise sample, averaged over several fresh draws each step.
The paper describes the result as returning the task to a random
initialization.

The averaging does something different. For zero-mean noise `z`, the expected
value of `||x - z||^2` is `||x||^2 + d`, which is smallest at `x = 0`. So
averaging over fresh draws drives the generated weights toward **zero**, not
toward noise. Measured over 3,000 steps on a 2,000-value vector:

| Noise strategy | Final spread of the weights |
| --- | --- |
| Averaged over 10 fresh draws each step (the paper) | 0.04 |
| One fresh draw each step | 0.07 |
| One fixed draw, reused every step | 0.98 |

Only the fixed draw lands on something noise-shaped. That variant is the
"Fixed-noise Alignment" the paper compares against in Appendix E and reports
as worse.

This does not break unlearning. A collapsed generator cannot classify, so
forget accuracy still lands at chance, which is what the metric asks for. But
it is worth knowing that the mechanism is weight collapse rather than
randomization, because the two differ in one way that could matter: a
collapsed network is identical for every forgotten task, while a randomized
one is not.

## Differences from the paper

| Here | Paper |
| --- | --- |
| Kaiming init with output scaling, and 1/fan-in on the classifier | Hyperfan initialization |
| Constant learning rate | Adam with a scheduler, unspecified |
| Permuted MNIST and Tiny ImageNet | Also 5-Tasks and CIFAR-100 |

Everything else follows the paper: ResNet18 or ResNet50 generated in 200
chunks, 32-number task and chunk codes, one head per parameter type, per-task
BatchNorm statistics, beta 0.1, gamma 0.01, 10 noise samples, and a burn-in of
100 annealed by 10% per unlearn down to a floor of 20.

The classifier is scaled by one over fan-in rather than He's square root of two
over fan-in. He is derived for a hidden layer feeding a ReLU; on a layer whose
outputs are logits it makes them too large. With He there, the `cnn` backbone
could not learn at all: first-epoch loss near 6.7, then pinned at ln(10) =
2.3026 with accuracy at exactly 10.0 for as many epochs as it was given, because
the early Adam steps overshoot into producing identical logits for every input
and never climb back out. One over fan-in starts the logits near zero instead.
This also changes the ResNet runs, where the effect was milder because batch
normalization keeps the features entering the classifier near unit scale.

Three settings the paper leaves open, decided here: how the 200 chunks are
shared between the heads (one each, then in proportion to size), what happens
to a forgotten task's BatchNorm statistics (nothing, since eq. 3 covers
generated parameters only), and which Tiny ImageNet classes form each task.

## Tiny ImageNet tasks

The paper says 10 tasks of 10 classes each, and 20 tasks of 10 classes for the
long 30-request run. It never says which classes go together or in what order,
so this had to be decided rather than read off.

The grouping lives in `uncle/task_partition.json`: the 200 WordNet IDs sorted,
then shuffled with NumPy PCG64 at seed 42, then cut into 20 disjoint groups of
ten. The file is written once and checked on every run, so a partition that
disagrees with it is an error rather than a silent change. The order inside a
group fixes the local labels 0 to 9.

Twenty tasks covers all 200 classes. All three of Table 4's Tiny-ImageNet
request sequences are in `uncle/config.py`: 30 requests each over tasks 0-19.
`--sequence` picks the row, and the task count and beta follow the dataset, so
`--dataset tiny_imagenet` gives 20 tasks and beta 0.01 without any other flag.

There used to be a second Tiny ImageNet loader with a different grouping, which
meant task 3 named different classes depending on which one you went through.
There is now one, and `uncle/data.py` routes to it.

With ResNet18 the heads come out as 195 chunks for ordinary weights, 4 for
residual connections and 1 for BatchNorm, and the hypernetwork is 56,082,990
parameters generating 11,172,810. The `cnn` backbone has no BatchNorm and no
residual connections, so it produces a single head.

## Citation

Adhikari, Kumaravelu, and Srijith (2025), [An Unlearning Framework for Continual
Learning](https://arxiv.org/abs/2509.17530). The BibTeX entry for the paper
reimplemented here is in [references.bib](references.bib).
