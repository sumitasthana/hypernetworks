"""The two operations: learn a task, forget a task.

The hypernetwork generates parameters. It does not generate buffers, and
BatchNorm's running mean and variance are buffers. So each task keeps its own
set, updated only while that task is being learned, and held still during
evaluation and forgetting.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .hypernet import HyperNetwork
from .telemetry import progress_bar


class UnCLe:
    """Runs learn and forget requests against one hypernetwork."""

    def __init__(
        self,
        hypernet: HyperNetwork,
        config: Config,
        target: nn.Module | None = None,
        tasks: dict[str, dict[str, Dataset]] | None = None,
        progress: bool = False,
    ):
        # target and tasks are only needed to measure accuracy and to learn.
        # A caller that just wants `forget` can leave them out.
        self.hypernet = hypernet
        self.target = target
        self.tasks = tasks
        self.config = config
        self.device = config.torch_device
        # One request can be forty minutes of ResNet50. A bar per request tells
        # you nothing while it runs, so the bars below count optimizer steps.
        self.progress = progress

        # A clean set of running statistics for every new task to start from.
        self.buffer_template: dict[str, torch.Tensor] = (
            {name: value.detach().clone() for name, value in target.named_buffers()}
            if target is not None
            else {}
        )
        self.task_buffers: dict[str, dict[str, torch.Tensor]] = {}

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

        # This task's own running statistics, updated only by this loop.
        self.task_buffers[task] = {
            name: value.clone() for name, value in self.buffer_template.items()
        }

        # Appendix B: chunk codes are learned by backpropagation and frozen
        # after the first task, to prevent catastrophic forgetting.
        first_task = len(self.hypernet.task_codes) == 1
        trainable = [*self.hypernet.generator_parameters(), code]
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
        self.target.train()
        epoch_losses = []

        bar = progress_bar(self.config.epochs * len(loader),
                           f"learn {task}", self.progress, leave=False)

        for epoch in range(self.config.epochs):
            running, batches = 0.0, 0

            for images, labels in loader:
                weights = self.hypernet.weights_for(task)
                # Passing this task's buffers lets BatchNorm update them here,
                # and only here.
                scores = functional_call(
                    self.target,
                    (weights, self.task_buffers[task]),
                    (images.to(self.device),),
                    strict=True,
                )

                loss = F.cross_entropy(scores, labels.to(self.device))
                loss = loss + self.config.beta * self.preserve(protected, snapshot)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running += loss.item()
                batches += 1
                bar.update(1)
                bar.set_postfix_str(
                    f"epoch {epoch + 1}/{self.config.epochs}, loss {loss.item():.3f}")

            epoch_losses.append(running / batches)

        bar.close()
        return epoch_losses

    # -- forget --------------------------------------------------------------

    def forget(
        self, task: str, protected: list[str], burn_in: int | None = None
    ) -> list[float]:
        """Teach the hypernetwork to turn one task's code into noise. Paper eq. 3.

        No data loader appears below. Forgetting needs only the task's code and
        a noise generator, which is what makes it usable in a continual setting
        where the task's data is long gone.

        This leaves the task's BatchNorm buffers alone, because eq. 3 covers
        generated parameters only. Those statistics are still derived from the
        forgotten task's data. See the README.
        """
        if task not in self.hypernet.task_codes:
            raise ValueError(f"Task {task} was never learned.")

        iterations = self.config.burn_in if burn_in is None else burn_in

        snapshot = self.hypernet.snapshot()
        trainable = self.hypernet.generator_parameters()

        self.hypernet.requires_grad_(False)
        for parameter in trainable:
            parameter.requires_grad_(True)

        optimizer = torch.optim.Adam(trainable, lr=self.config.learning_rate)
        self.hypernet.train()
        losses = []

        bar = progress_bar(iterations, f"forget {task}", self.progress, leave=False)

        for _ in range(iterations):
            raw = self.hypernet.raw_for(task)

            # The paper averages over fresh draws so the hypernetwork cannot
            # memorize one noise sample. Because the draws are zero-mean, the
            # average squared distance is ||raw||^2 + d, which is smallest at
            # zero. So this collapses the generated weights toward zero rather
            # than randomizing them. Either way the task stops working. See
            # "What the noise objective actually does" in the README.
            to_noise = sum(
                (raw - torch.randn_like(raw)).square().sum()
                for _ in range(self.config.noise_samples)
            ) / self.config.noise_samples

            loss = self.config.gamma * to_noise + self.preserve(protected, snapshot)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            bar.update(1)
            bar.set_postfix_str(f"loss {loss.item():.3f}")

        bar.close()
        return losses

    # -- measurement ---------------------------------------------------------

    @torch.no_grad()
    def accuracy(self, task: str, split: str = "test") -> float:
        """Accuracy of the network this task's code currently produces."""
        self.hypernet.eval()
        self.target.eval()

        weights = self.hypernet.weights_for(task)
        # Copies, so measuring cannot disturb the stored statistics.
        buffers = {
            name: value.clone() for name, value in self.task_buffers[task].items()
        }

        loader = DataLoader(
            self.tasks[task][split], batch_size=self.config.eval_batch_size
        )

        correct, total = 0, 0
        for images, labels in loader:
            scores = functional_call(
                self.target, (weights, buffers), (images.to(self.device),), strict=True
            )
            correct += (scores.argmax(1) == labels.to(self.device)).sum().item()
            total += labels.numel()

        return 100.0 * correct / total
