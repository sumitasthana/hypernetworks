"""The two operations: learn a task, forget a task."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .hypernet import HyperNetwork


class UnCLe:
    """Runs learn and forget requests against one hypernetwork."""

    def __init__(
        self,
        hypernet: HyperNetwork,
        target: nn.Module,
        tasks: dict[str, dict[str, Dataset]],
        config: Config,
    ):
        self.hypernet = hypernet
        self.target = target
        self.tasks = tasks
        self.config = config
        self.device = config.torch_device

    # -- the shared regularizer ---------------------------------------------

    def preserve(
        self, protected: list[str], snapshot: HyperNetwork
    ) -> torch.Tensor:
        """Hold other tasks' weights where the snapshot left them. Paper eq. 2.

        This one term is what stops a learn from wrecking old tasks and a
        forget from spilling onto its neighbours. Both operations pay it.
        Forgotten tasks stay on the protected list, which is what stops them
        coming back as later tasks are learned.
        """
        if not protected:
            return torch.zeros((), device=self.device)

        total = torch.zeros((), device=self.device)
        for task in protected:
            code = self.hypernet.task_codes[task].detach()
            with torch.no_grad():
                before = snapshot.weights_from_code(code)
            now = self.hypernet.weights_from_code(code)
            total = total + sum(
                (now[name] - before[name]).square().sum() for name in now
            )

        return total / len(protected)

    # -- learn ---------------------------------------------------------------

    def learn(self, task: str, protected: list[str]) -> list[float]:
        """Teach the hypernetwork to classify one task. Returns loss per epoch."""
        snapshot = self.hypernet.snapshot()
        code = self.hypernet.add_task(task)

        # Chunk codes are learned on the first task only, then left alone.
        first_task = len(self.hypernet.task_codes) == 1
        trainable = [*self.hypernet.trunk.parameters(), code]
        if first_task:
            trainable.append(self.hypernet.chunk_codes)

        self.hypernet.requires_grad_(False)
        for parameter in trainable:
            parameter.requires_grad_(True)

        optimizer = torch.optim.Adam(trainable, lr=self.config.learning_rate)
        loader = DataLoader(
            self.tasks[task]["train"],
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.config.seed),
        )

        self.hypernet.train()
        epoch_losses = []

        for _ in range(self.config.epochs):
            running, batches = 0.0, 0

            for images, labels in loader:
                weights = self.hypernet.weights_for(task)
                scores = functional_call(self.target, weights, (images.to(self.device),))

                loss = F.cross_entropy(scores, labels.to(self.device))
                loss = loss + self.config.beta * self.preserve(protected, snapshot)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running += loss.item()
                batches += 1

            epoch_losses.append(running / batches)

        return epoch_losses

    # -- forget --------------------------------------------------------------

    def forget(self, task: str, protected: list[str]) -> list[float]:
        """Teach the hypernetwork to turn one task's code into noise. Paper eq. 3.

        No data loader appears below. Forgetting needs only the task's code and
        a noise generator, which is what makes it usable in a continual setting
        where the task's data is long gone.
        """
        if task not in self.hypernet.task_codes:
            raise ValueError(f"Task {task} was never learned.")

        snapshot = self.hypernet.snapshot()
        trainable = list(self.hypernet.trunk.parameters())

        self.hypernet.requires_grad_(False)
        for parameter in trainable:
            parameter.requires_grad_(True)

        optimizer = torch.optim.Adam(trainable, lr=self.config.learning_rate)
        self.hypernet.train()
        losses = []

        for _ in range(self.config.burn_in):
            raw = self.hypernet.raw_for(task)

            # Aim at several noise draws at once, so the hypernetwork learns
            # "be noise" rather than memorizing one particular noise sample.
            to_noise = sum(
                (raw - torch.randn_like(raw)).square().sum()
                for _ in range(self.config.noise_samples)
            ) / self.config.noise_samples

            loss = self.config.gamma * to_noise + self.preserve(protected, snapshot)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        return losses

    # -- measurement ---------------------------------------------------------

    @torch.no_grad()
    def accuracy(self, task: str, split: str = "test") -> float:
        """Accuracy of the network this task's code currently produces."""
        self.hypernet.eval()
        weights = self.hypernet.weights_for(task)
        loader = DataLoader(
            self.tasks[task][split], batch_size=self.config.eval_batch_size
        )

        correct, total = 0, 0
        for images, labels in loader:
            scores = functional_call(self.target, weights, (images.to(self.device),))
            correct += (scores.argmax(1) == labels.to(self.device)).sum().item()
            total += labels.numel()

        return 100.0 * correct / total
