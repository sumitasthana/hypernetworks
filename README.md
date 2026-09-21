# UnCLe on Permuted MNIST and Tiny ImageNet

A reimplementation of [An Unlearning Framework for Continual
Learning](https://arxiv.org/abs/2509.17530) (Adhikari, Kumaravelu, Srijith).

One hypernetwork produces the weights of a small CNN from a short code, one
code per task. Learning a task trains the hypernetwork to classify it.
Forgetting a task trains the hypernetwork to turn that task's code into noise,
which needs no data at all.

## Run it

The defaults are the paper's Permuted-MNIST setting: ResNet18 generated in 200
chunks, 10 tasks, and request sequence 1 from Table 4. That wants a GPU.

```bash
pip install -r requirements.txt
python main.py                     # the paper's setting
python main.py --sequence 2        # sequences 1, 2 and 3 are all from Table 4
```

For a quick check without a GPU, swap in the small stand-in network:

```bash
python main.py --backbone cnn --chunks 32 --epochs 1
python tests/test_uncle.py         # fourteen checks, seconds, no download
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

| File | What it holds |
| --- | --- |
| `uncle/config.py` | Every knob, plus validation of the request list |
| `uncle/data.py` | Permuted MNIST and Tiny ImageNet task streams |
| `uncle/hypernet.py` | The target CNN and the network that generates its weights |
| `uncle/trainer.py` | `learn`, `forget`, and the regularizer they share |
| `uncle/metrics.py` | Retain accuracy, forget accuracy, spill, relapse |
| `uncle/experiment.py` | Works through a request sequence, returns records |
| `main.py` | Command line entry point and printing |
| `uncle_minimal.py` | The same method in one flat file, for reading |
| `uncle_from_scratch/` | A separate rebuild, data first, reusing this package |
| `Scalable_Hypernetworks_...ipynb` | Colab notebook, a scaled-up run in progress |

`uncle_minimal.py` is not imported by anything. It exists so the method can be
read top to bottom in one sitting before meeting the package.

## The rebuild next door

`uncle_from_scratch/` works up to the same method in stages, starting from the
data rather than the method, so each piece can be checked before the next one
lands. It has its own Tiny ImageNet loader and its own task partition: 200
WordNet IDs shuffled with seed 42 into 20 groups of ten, saved to a file and
verified on every run.

It does not reimplement the method. The hypernetwork, the learn and forget
operations, the four metrics and the Table 4 sequences all come from this
package. Three modules join the two, and they are libraries first because the
real runs happen on a Colab GPU:

```python
import sys
sys.path.insert(0, "uncle_from_scratch")

from run import run_experiment
history, numbers = run_experiment(sequence=1, backbone="resnet50", epochs=5)
```

Any `Config` field passes through as a keyword argument, so a sweep is a loop
over calls. `uncle_from_scratch/colab_guide.html` is the Colab walkthrough:
nine cells and five worked examples. `uncle_from_scratch/README.md` has the
rest.

Task IDs are not comparable between the two. `uncle/data.py` cuts sorted
WordNet IDs into consecutive blocks; the rebuild shuffles them first. Task 3 is
ten different classes in each, so name the partition whenever you compare.

## Use it from your own code

```python
from uncle import Config, run, summary

history = run(Config(epochs=3, requests=(("learn", "A"), ("learn", "B"), ("forget", "A"))))
print(summary(history))
```

`run` takes an optional `on_request` callback, called with each record as it
completes, so you can log or plot without changing the library.

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
| Kaiming init with output scaling | Hyperfan initialization |
| Constant learning rate | Adam with a scheduler, unspecified |
| Permuted MNIST and Tiny ImageNet | Also 5-Tasks and CIFAR-100 |

Everything else follows the paper: ResNet18 or ResNet50 generated in 200
chunks, 32-number task and chunk codes, one head per parameter type, per-task
BatchNorm statistics, beta 0.1, gamma 0.01, 10 noise samples, and a burn-in of
100 annealed by 10% per unlearn down to a floor of 20.

Three settings the paper leaves open, decided here: how the 200 chunks are
shared between the heads (one each, then in proportion to size), what happens
to a forgotten task's BatchNorm statistics (nothing, since eq. 3 covers
generated parameters only), and which Tiny ImageNet classes form each task.

## Tiny ImageNet tasks

The paper says 10 tasks of 10 classes each, and 20 tasks of 10 classes for the
long 30-request run. It never says which classes go together or in what order,
so that is a knob here rather than a claim:

```bash
python main.py --dataset tiny_imagenet --class-order sorted   # default
python main.py --dataset tiny_imagenet --class-order random   # uses --seed
```

`sorted` cuts the 200 wnids into consecutive blocks of ten in alphabetical
order, so task 0 is classes 0-9. `random` shuffles them first. The groups are
always disjoint. Twenty tasks covers all 200 classes; drop to ten tasks in
`Config` and only the first 100 are used.

All three of Table 4's Tiny-ImageNet request sequences are in
`uncle/config.py`: 30 requests each over tasks 0-19. `--sequence` picks the
row, and the task count and beta follow the dataset, so `--dataset
tiny_imagenet` gives 20 tasks and beta 0.01 without any other flag.

With ResNet18 the heads come out as 195 chunks for ordinary weights, 4 for
residual connections and 1 for BatchNorm, and the hypernetwork is 56,082,990
parameters generating 11,172,810. The `cnn` backbone has no BatchNorm and no
residual connections, so it produces a single head.
