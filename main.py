"""Run one UnCLe experiment and print the paper's four numbers.

The defaults are the paper's Permuted-MNIST setting and need a GPU:

    python main.py                          # ResNet18, 200 chunks, 15 requests

For a quick check on a CPU, use the small stand-in network:

    python main.py --backbone cnn --chunks 32 --epochs 1 --sequence 1
"""

import argparse

from uncle import Config, run, spill, summary
from uncle.config import PERMUTED_MNIST_SEQUENCES, parse_sequence


def show(record: dict) -> None:
    """Print one line per completed request."""
    line = f"{record['action']:6} {record['task']}  " + "  ".join(
        f"{task}={record['after'][task]:5.1f}%" for task in record["seen"]
    )

    if record["action"] == "forget":
        line += f"    spill={spill(record):6.2f}  burn-in={record['burn_in']}"

    print(line)


def parse_args() -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--backbone", default=defaults.backbone,
                        choices=("resnet18", "resnet50", "cnn"))
    parser.add_argument("--sequence", type=int, default=1, choices=(1, 2, 3),
                        help="Which request sequence from the paper's Table 4")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--chunks", type=int, default=defaults.chunks)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--beta", type=float, default=defaults.beta)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--burn-in", type=int, default=defaults.burn_in)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)

    arguments = parser.parse_args()
    return Config(
        requests=parse_sequence(PERMUTED_MNIST_SEQUENCES[arguments.sequence]),
        backbone=arguments.backbone,
        epochs=arguments.epochs,
        chunks=arguments.chunks,
        batch_size=arguments.batch_size,
        beta=arguments.beta,
        gamma=arguments.gamma,
        burn_in=arguments.burn_in,
        seed=arguments.seed,
        device=arguments.device,
    )


def main() -> None:
    config = parse_args()

    print(f"backbone: {config.backbone}   chunks: {config.chunks}   "
          f"epochs: {config.epochs}   device: {config.device}")
    print("requests: " + " ".join(
        f"{action[0].upper()}{task}" for action, task in config.requests
    ))
    print()

    history = run(config, on_request=show)
    numbers = summary(history)

    print(f"\nRetain accuracy : {numbers['retain_accuracy']:.2f}%   (higher is better)")
    print(f"Forget accuracy : {numbers['forget_accuracy']:.2f}%   (chance is 10%)")
    print(f"Mean spill      : {numbers['mean_spill']:.3f}    (lower is better)")
    print(f"Mean relapse    : {numbers['mean_relapse']:.3f}    (lower is better)")


if __name__ == "__main__":
    main()
