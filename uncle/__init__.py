"""UnCLe: an unlearning framework for continual learning.

Paper: An Unlearning Framework for Continual Learning (arXiv 2509.17530).

One hypernetwork produces the weights of a target network from a short code,
one code per task. Learning a task trains the hypernetwork to classify it.
Forgetting a task trains the hypernetwork to turn that task's code into noise,
which needs no data at all.
"""

from .baseline import run_baseline
from .config import Config
from .diagnostics import diagnose_forgetting
from .experiment import run
from .experiments import compare, make_config, run_experiment, run_sequences
from .hypernet import HyperNetwork, build_target
from .metrics import relapse, spill, summary
from .streams import build_tasks
from .trainer import UnCLe

__all__ = [
    "Config",
    "HyperNetwork",
    "UnCLe",
    "build_target",
    "build_tasks",
    "compare",
    "diagnose_forgetting",
    "make_config",
    "relapse",
    "run",
    "run_baseline",
    "run_experiment",
    "run_sequences",
    "spill",
    "summary",
]
