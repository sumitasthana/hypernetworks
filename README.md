# UnCLe on Permuted MNIST

A reimplementation of [An Unlearning Framework for Continual
Learning](https://arxiv.org/abs/2509.17530) (Adhikari, Kumaravelu, Srijith).

One hypernetwork produces the weights of a small CNN from a short code, one
code per task. Learning a task trains the hypernetwork to classify it.
Forgetting a task trains the hypernetwork to turn that task's code into noise,
which needs no data at all.

## Run it

```bash
pip install -r requirements.txt
python main.py
```

About two minutes on a CPU. Then:

```bash
python tests/test_uncle.py      # seven checks, seconds, no download
python main.py --epochs 3 --chunks 64
```

## What is where

| File | What it holds |
| --- | --- |
| `uncle/config.py` | Every knob, plus validation of the request list |
| `uncle/data.py` | Permuted MNIST, one fixed pixel shuffle per task |
| `uncle/hypernet.py` | The target CNN and the network that generates its weights |
| `uncle/trainer.py` | `learn`, `forget`, and the regularizer they share |
| `uncle/metrics.py` | Retain accuracy, forget accuracy, spill, relapse |
| `uncle/experiment.py` | Works through a request sequence, returns records |
| `main.py` | Command line entry point and printing |
| `uncle_minimal.py` | The same method in one flat file, for reading |
| `Scalable_Hypernetworks_...ipynb` | Colab notebook, a scaled-up run in progress |

`uncle_minimal.py` is not imported by anything. It exists so the method can be
read top to bottom in one sitting before meeting the package.

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

## Differences from the paper

| Here | Paper |
| --- | --- |
| 4-layer CNN, 20k weights | ResNet18 or ResNet50 |
| No normalization layers | BatchNorm, which needs per-task running statistics |
| One output head | Separate heads per parameter group |
| Kaiming init with output scaling | Hyperfan initialization |
| 3 tasks, 4 requests | Up to 20 tasks, 30 requests |

The mechanism is the same. The scale is not, so treat the numbers here as a
demonstration that it works, not as a reproduction of the paper's results.
