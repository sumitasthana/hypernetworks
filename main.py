"""Run one UnCLe experiment and print the paper's four numbers.

    python main.py
    python main.py --epochs 3 --chunks 64
"""

import argparse

from uncle import Config, run, spill, summary


def show(record: dict) -> None:
    """Print one line per completed request."""
    line = f"{record['action']:6} {record['task']}  " + "  ".join(
        f"{task}={record['after'][task]:5.1f}%" for task in record["seen"]
    )

    if record["action"] == "forget":
        line += f"    spill={spill(record):5.2f}"

    print(line)


def parse_args() -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--chunks", type=int, default=defaults.chunks)
    parser.add_argument("--beta", type=float, default=defaults.beta)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--burn-in", type=int, default=defaults.burn_in)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)

    arguments = parser.parse_args()
    return Config(
        epochs=arguments.epochs,
        chunks=arguments.chunks,
        beta=arguments.beta,
        gamma=arguments.gamma,
        burn_in=arguments.burn_in,
        seed=arguments.seed,
        device=arguments.device,
    )


def main() -> None:
    config = parse_args()
    print(f"device: {config.device}   requests: "
          + " ".join(f"{action[0].upper()}{task}" for action, task in config.requests))

    history = run(config, on_request=show)
    numbers = summary(history)

    print(f"\nRetain accuracy : {numbers['retain_accuracy']:.2f}%   (higher is better)")
    print(f"Forget accuracy : {numbers['forget_accuracy']:.2f}%   (chance is 10%)")
    print(f"Mean spill      : {numbers['mean_spill']:.3f}    (lower is better)")
    print(f"Mean relapse    : {numbers['mean_relapse']:.3f}    (lower is better)")


if __name__ == "__main__":
    main()
