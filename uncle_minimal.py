"""UnCLe on Permuted MNIST, the whole method in one file.

Paper: An Unlearning Framework for Continual Learning (arXiv 2509.17530).

The idea in three sentences:
  One small network, the hypernetwork, produces the weights of a CNN.
  It decides what to produce from a short "task code", one code per task.
  To forget a task, retrain the hypernetwork so that task's code produces noise.

Nothing here is data-free by accident. Forgetting never touches the task's data,
because all it needs is the task code and a noise generator.

One deliberate departure from the paper: Appendix B splits the hypernetwork's
last layer into one head per parameter type. This file keeps a single head,
because the CNN below has no BatchNorm and no residual connections, so all its
parameters fall into one type anyway. The uncle package implements the split.

Run it:  python uncle_minimal.py
"""

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

SEED = 0
TASKS = ["A", "B", "C"]
REQUESTS = [("learn", "A"), ("learn", "B"), ("forget", "A"), ("learn", "C")]

EPOCHS = 1
BATCH = 128
LR = 1e-3

BETA = 0.1       # how hard we hold other tasks still while learning (paper eq. 2)
GAMMA = 0.01     # how hard we push the forgotten task toward noise (paper eq. 3)

CODE_DIM = 32    # length of a task code, and of a chunk code
CHUNKS = 32      # the CNN's weights come out in this many slices
NOISE_SAMPLES = 10
BURN_IN = 100    # optimizer steps spent on one forget request

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

# ----------------------------------------------------------------------------
# Data: every task is MNIST with its pixels shuffled a different way
# ----------------------------------------------------------------------------

to_tensor = transforms.ToTensor()
mnist_train = datasets.MNIST("./data", train=True, download=True, transform=to_tensor)
mnist_test = datasets.MNIST("./data", train=False, download=True, transform=to_tensor)

permutations = {task: torch.randperm(28 * 28) for task in TASKS}


class Permuted(Dataset):
    """Same digits, pixels rearranged by one fixed shuffle."""

    def __init__(self, base, permutation):
        self.base = base
        self.permutation = permutation

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return image.reshape(-1)[self.permutation].reshape(1, 28, 28), label


# ----------------------------------------------------------------------------
# The CNN. It owns no trained weights: every forward pass gets weights handed in.
# ----------------------------------------------------------------------------

target = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(32 * 7 * 7, 10),
).to(DEVICE).requires_grad_(False)

# Where each weight tensor sits inside one long flat vector, and how much to
# shrink it. The shrink factors turn unit-spread raw numbers into a textbook
# He initialization, so the generated CNN starts out correctly scaled.
layout, scales, offset = {}, {}, 0
for name, parameter in target.named_parameters():
    layout[name] = (offset, parameter.shape)
    fan_in = parameter[0].numel() if parameter.dim() > 1 else 1
    scales[name] = (2 / fan_in) ** 0.5 if parameter.dim() > 1 else 0.01
    offset += parameter.numel()

TOTAL = offset
WIDTH = -(-TOTAL // CHUNKS)  # numbers each chunk has to produce, rounded up

# ----------------------------------------------------------------------------
# The hypernetwork
# ----------------------------------------------------------------------------

hypernet = nn.Sequential(
    nn.Linear(2 * CODE_DIM, 128), nn.ReLU(),
    nn.Linear(128, 256), nn.ReLU(),
    nn.Linear(256, 512), nn.ReLU(),
    nn.Linear(512, WIDTH),
).to(DEVICE)

for layer in hypernet:
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        nn.init.zeros_(layer.bias)

# Last layer aims for raw outputs with spread near 1, so that after the shrink
# factors above the generated CNN starts out correctly initialized.
nn.init.normal_(hypernet[-1].weight, std=512 ** -0.5)

chunk_codes = nn.Parameter(torch.randn(CHUNKS, CODE_DIM, device=DEVICE))
task_codes = {}


def raw_from(task_code, net, chunks):
    """One task code in, one flat vector of unscaled numbers out."""
    # Pair the task code with every chunk code, then run all chunks at once.
    pairs = torch.cat([task_code.expand(CHUNKS, -1), chunks], dim=1)
    return net(pairs).reshape(-1)[:TOTAL]


def weights_from(task_code, net, chunks):
    """One task code in, a full set of CNN weights out."""
    raw = raw_from(task_code, net, chunks)
    return {
        name: raw[start:start + shape.numel()].reshape(shape) * scales[name]
        for name, (start, shape) in layout.items()
    }


# ----------------------------------------------------------------------------
# The two operations
# ----------------------------------------------------------------------------

def preserve(protected, snapshot, snapshot_chunks):
    """Hold other tasks' weights where the snapshot left them. Paper eq. 2.

    This one term is what stops a learn from wrecking old tasks and a forget
    from spilling onto its neighbours. Both operations pay it.
    """
    if not protected:
        return torch.zeros((), device=DEVICE)

    total = torch.zeros((), device=DEVICE)
    for task in protected:
        code = task_codes[task].detach()
        with torch.no_grad():
            before = weights_from(code, snapshot, snapshot_chunks)
        now = weights_from(code, hypernet, chunk_codes)
        total = total + sum((now[n] - before[n]).square().sum() for n in now)

    return total / len(protected)


def take_snapshot():
    """Freeze the hypernetwork as it is now, to compare against later."""
    return copy.deepcopy(hypernet).requires_grad_(False), chunk_codes.detach().clone()


def learn(task, protected):
    """Teach the hypernetwork to classify one task."""
    snapshot, snapshot_chunks = take_snapshot()

    task_codes[task] = nn.Parameter(torch.randn(CODE_DIM, device=DEVICE))
    trainable = list(hypernet.parameters()) + [task_codes[task]]
    if len(task_codes) == 1:
        trainable.append(chunk_codes)  # Appendix B: learned on task one, then frozen
    optimizer = torch.optim.Adam(trainable, lr=LR)

    loader = DataLoader(
        Permuted(mnist_train, permutations[task]), batch_size=BATCH, shuffle=True
    )

    for _ in range(EPOCHS):
        for images, labels in loader:
            weights = weights_from(task_codes[task], hypernet, chunk_codes)
            scores = functional_call(target, weights, (images.to(DEVICE),))

            loss = F.cross_entropy(scores, labels.to(DEVICE))
            loss = loss + BETA * preserve(protected, snapshot, snapshot_chunks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def forget(task, protected):
    """Teach the hypernetwork to turn one task's code into noise. Paper eq. 3.

    No data loader appears anywhere below. That is the point.
    """
    snapshot, snapshot_chunks = take_snapshot()
    optimizer = torch.optim.Adam(hypernet.parameters(), lr=LR)

    for _ in range(BURN_IN):
        raw = raw_from(task_codes[task].detach(), hypernet, chunk_codes)

        # The paper averages over fresh draws so the hypernetwork cannot
        # memorize one noise sample. Because the draws are zero-mean, this
        # actually drives the weights toward zero rather than toward noise.
        # The task stops working either way. See the README.
        to_noise = sum(
            (raw - torch.randn_like(raw)).square().sum() for _ in range(NOISE_SAMPLES)
        ) / NOISE_SAMPLES

        loss = GAMMA * to_noise + preserve(protected, snapshot, snapshot_chunks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def accuracy(task):
    """Test accuracy of the CNN this task's code currently produces."""
    weights = weights_from(task_codes[task], hypernet, chunk_codes)
    loader = DataLoader(Permuted(mnist_test, permutations[task]), batch_size=512)

    correct = 0
    for images, labels in loader:
        scores = functional_call(target, weights, (images.to(DEVICE),))
        correct += (scores.argmax(1) == labels.to(DEVICE)).sum().item()

    return 100.0 * correct / len(mnist_test)


# ----------------------------------------------------------------------------
# Run the sequence
# ----------------------------------------------------------------------------

def main():
    print(f"CNN weights: {TOTAL:,}   hypernetwork: "
          f"{sum(p.numel() for p in hypernet.parameters()):,}   device: {DEVICE}")

    seen = []
    for action, task in REQUESTS:
        before = {t: accuracy(t) for t in seen}

        # Every task met so far, except the one this request is about.
        # Forgotten tasks stay on the list, which is what stops them relapsing.
        protected = [t for t in seen if t != task]

        if action == "learn":
            learn(task, protected)
            seen.append(task)
        else:
            forget(task, protected)

        after = {t: accuracy(t) for t in seen}
        line = f"{action:6} {task}  " + "  ".join(f"{t}={after[t]:5.1f}%" for t in seen)
        if action == "forget":
            spill = sum(abs(after[t] - before[t]) for t in before if t != task)
            line += f"    spill={spill:5.2f}"
        print(line)

    final = {task: accuracy(task) for task in seen}
    assert final["A"] < 20, f"A was forgotten but still scores {final['A']:.1f}%"
    assert final["B"] > 80, f"B should have survived, scores {final['B']:.1f}%"
    assert final["C"] > 80, f"C was learned last, scores {final['C']:.1f}%"
    print("\nChecks passed: A is back at chance, B and C are intact.")


if __name__ == "__main__":
    main()
