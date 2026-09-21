# How this was built, stage by stage

A record of the staged rebuild. The code it describes now lives in `uncle/`;
paths below point at where each piece ended up.

This was built as a separate, incremental implementation and has since been merged into `uncle/`. The first
stage loads and explores Tiny ImageNet. Model training, hypernetworks, and
unlearning will build on this stage later.

Start with `notebooks/01_tiny_imagenet.ipynb` and run the cells in order. It walks through
the directory layout, class IDs, one image, split counts, examples, class balance,
sampled pixel statistics, and a PyTorch batch.

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m jupyterlab notebooks/01_tiny_imagenet.ipynb
```

For the same exploration as a script:

```powershell
python scripts/explore.py
```

It reuses `data/tiny-imagenet-200` and saves example images, a class
balance plot, and `summary.json` in `outputs/`. These are measurements from the
selected dataset. To use another extracted copy, pass `--root PATH`. To download
the official archive when the dataset directory does not exist, add `--download`.
The notebook exposes the same choice as `DOWNLOAD`.

The [Stanford dataset description](https://cs231n.stanford.edu/reports/2015/pdfs/yle_project.pdf)
specifies 200 classes with 500 training and 50 validation images per class, at
64 by 64 pixels. The released test images have no labels. We keep validation
named `val` and do not report it as a labeled test set.

`uncle/tinyimagenet.py` reads validation labels from `val_annotations.txt`, uses one sorted
WordNet-ID mapping across splits, and converts images to RGB when loaded. It
supports the original archive and class folders created by the older loader
without moving any files. `words.txt` supplies readable names; only IDs from
`wnids.txt` define this dataset's classes.

`scripts/explore.py` uses a fixed seed and estimates RGB statistics from 2,048 training
images by default. Those estimates describe pixels scaled to [0, 1]; no
augmentation or normalization is applied. Counts cover the indexed splits,
but image decoding checks cover only the sampled images and displayed batch.

## Task 1

Continue with `notebooks/02_task1.ipynb`, or prepare the task from the repository root:

```powershell
python scripts/task1.py
```

Task IDs are **zero-based**: task ID `1` is the second group, following task `0`.
`uncle/tasks.py` sorts the 200 WordNet IDs, shuffles them using NumPy PCG64 with seed
42, and divides them into 20 disjoint groups of ten. The full assignment is
saved in `uncle/task_partition.json` and checked on subsequent runs. A conflicting
assignment raises an error instead of overwriting the file.

The order of IDs within each group defines local labels 0 through 9. Task 1
contains 5,000 training and 500 validation images in the complete dataset.
Its `ClassTask` wrappers select images without copying them. Training batches
shuffle with a seeded generator; validation batches keep their order. Both use
RGB float tensors in [0, 1] and the same local labels.

`scripts/task1.py` saves the measured class mapping, counts, batch shapes, and an example
grid under `outputs/task1/`. The task loader is also available directly when
working from this folder:

```python
from tasks import build_task

train, val, train_loader, val_loader = build_task(task_id=1)
images, labels = next(iter(train_loader))
# images: [32, 3, 64, 64]; labels: [32], values from 0 through 9
```

Run the partition and label-mapping checks with:

```powershell
python tests/test_tasks.py
```

## The experiment

The tasks above are this folder's own work. The method is reused from the
sibling `uncle` package rather than written twice: the hypernetwork, the
learn and forget operations, the four metrics, and the paper's Table 4 request
sequences all live there and are already tested.

Three modules join the two:

- `uncle/streams.py` turns the saved partition into the `{task: {"train", "test"}}`
  datasets the trainer wants. Tiny ImageNet's public test split has no labels,
  so the `test` slot holds `val`: every accuracy here is validation accuracy.
- `uncle/baseline.py` trains the target network on one task with ordinary backprop.
  No hypernetwork. It answers whether the architecture can learn a task at all,
  which is worth knowing before debugging anything harder.
- `uncle/experiments.py` works through a request sequence and reports the four numbers.

Both are libraries first and command line scripts second, because the real runs
happen on a Colab GPU. Every knob is a keyword argument, and any `Config` field
can be passed straight through, so a sweep is a loop over calls.

### From a notebook

```python
!git clone https://github.com/<your-remote> repo
!pip install -q -r repo/requirements.txt

import sys
sys.path.insert(0, "/content/hypernetworks")

from run import make_config, run_experiment

# Downloads Tiny ImageNet on first call, about 240 MB.
history, numbers = run_experiment(sequence=1, backbone="resnet50",
                                  epochs=5, download=True)
```

`run_experiment` returns the history and the four numbers, so nothing has to be
read back off disk. Useful arguments:

| Argument | What it does |
| --- | --- |
| `sequence` | 1, 2 or 3: the row of the paper's Table 4 |
| `limit_requests` | Stop after the first n requests |
| `max_images` | Cap images per task, for a quick check |
| `download` | Fetch the dataset when it is missing |
| `output` | Directory for the JSON files, or `None` to keep it all in memory |
| `on_request` | Called with each record as it finishes, for live plots |
| `config` | A prepared `Config`, instead of the arguments above |
| anything else | Passed to `Config`: `epochs`, `chunks`, `beta`, `gamma`, `seed`, `device`, ... |

Sweeping a parameter:

```python
results = {}
for gamma in (0.001, 0.01, 0.1):
    _, results[gamma] = run_experiment(sequence=1, backbone="resnet18",
                                       gamma=gamma, output=None, verbose=False)
```

Inspect the settings before spending GPU hours on them:

```python
config = make_config(sequence=1, backbone="resnet50", epochs=5)
print(config)
history, numbers = run_experiment(config=config, output="/content/drive/MyDrive/uncle")
```

The baseline works the same way:

```python
from baseline import run_baseline
report = run_baseline(task=1, backbone="resnet50", epochs=10, download=True)
```

When `output` is set, the history is rewritten after every request, so a
thirty-request ResNet50 run can be watched as it goes and survives a
disconnected runtime. Point `output` at a mounted Drive folder to keep it.

### From the command line

```powershell
python scripts/baseline.py --task 1 --epochs 5
python scripts/run.py --backbone cnn --chunks 32 --epochs 1 --limit-requests 5 --max-images 300
python scripts/run.py --sequence 1 --backbone resnet50
```

The first two are CPU checks. The last is the paper's setting, 30 requests over
20 tasks with ResNet50 generated in 200 chunks, and wants a GPU and hours.

Run the end-to-end check with:

```powershell
python tests/test_experiments.py
```

Nine checks, about two minutes on a CPU. They cover the request loop and its
records, the metrics matching those records, capping keeping every class,
forgetting measured as the collapse of the generated weights, and learning
getting off chance. No training results are claimed at this stage.

Task IDs here are not comparable with `uncle`. That package cuts sorted
WordNet IDs into consecutive blocks; `uncle/tasks.py` shuffles them with seed 42
first. Task 3 is ten different classes in each, so say which partition produced
any number you compare.
