"""T0.1 from PLAN.md: is the collapse a matter of tuning, or is it structural?

One full run of sequence 1 returned 10% retain accuracy and a mean spill of
30.72 against the paper's 0.722. Forgetting works and takes everything else
with it. The forget objective is `gamma * noise + preserve`, so gamma sets how
hard forgetting pushes against the term that holds the other tasks still. This
probe varies gamma over the first three requests and watches what the one
unlearn does to the one retained task.

Three requests is the smallest window that can show it: sequence 1 opens with
learn 3, learn 0, forget 3. Task 3 is the target, task 0 is the bystander, and
the question is whether any gamma drives the first to chance without dragging
the second down with it.

    python scripts/gamma_probe.py --check     # validate, run nothing
    python scripts/gamma_probe.py             # the probe itself

Reading the result, per PLAN.md:

  a gamma passes  ->  tuning. Take it to T0.3 and re-run the reference.
  none passes     ->  structural. Go to T0.2 and look at the objective.

Each call uses `output=None` on purpose. Runs that differ only in gamma share a
file name, and resuming refuses a checkpoint whose config differs, so writing
them to one directory would fail on the second call. This script keeps the
results itself.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uncle.experiments import make_config, run_experiment
from uncle.tinyimagenet import DEFAULT_ROOT

#: PLAN.md T0.1. The target has to reach chance and the bystander has to stay.
TARGET_AT_MOST = 12.0        # per cent, task 3 after its forget
BYSTANDER_DRIFT = 5.0        # percentage points task 0 may move


def probe_one(gamma, *, backbone, epochs, root, download, max_images, verbose):
    """Run learn 3, learn 0, forget 3 at one gamma. Returns what it did."""
    started = time.perf_counter()
    result = run_experiment(
        sequence=1, limit_requests=3, gamma=gamma,
        backbone=backbone, epochs=epochs, root=root, download=download,
        max_images=max_images, output=None, verbose=verbose, progress=verbose,
    )
    history = result["history"]
    forget = history[2]

    # history[0] is learn 3, history[1] is learn 0, history[2] is forget 3.
    assert forget["action"] == "forget" and forget["task"] == "3", history
    return {
        "gamma": gamma,
        "target_before": forget["before"]["3"],
        "target_after": forget["after"]["3"],
        "bystander_before": forget["before"]["0"],
        "bystander_after": forget["after"]["0"],
        "bystander_drift": abs(forget["after"]["0"] - forget["before"]["0"]),
        "mean_spill": result["numbers"]["mean_spill"],
        "forget_loss": forget["final_loss"],
        "seconds": time.perf_counter() - started,
    }


def passes(row):
    return (row["target_after"] <= TARGET_AT_MOST
            and row["bystander_drift"] < BYSTANDER_DRIFT)


def table(rows):
    header = (f"{'gamma':>8} {'task 3 after':>13} {'task 0 before':>14} "
              f"{'task 0 after':>13} {'drift':>7} {'spill':>8} {'verdict':>9}")
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['gamma']:>8} {row['target_after']:>12.1f}% "
            f"{row['bystander_before']:>13.1f}% {row['bystander_after']:>12.1f}% "
            f"{row['bystander_drift']:>7.1f} {row['mean_spill']:>8.2f} "
            f"{'pass' if passes(row) else 'fail':>9}")
    return "\n".join(lines)


def verdict(rows):
    winners = [row for row in rows if passes(row)]
    if not winners:
        return ("STRUCTURAL. No gamma forgets task 3 without damaging task 0. "
                "Tuning will not save this; go to T0.2 and look at the balance "
                "between the noise and preservation terms, the burn-in length, "
                "and whether the noise target should be the scaled parameters "
                "rather than the raw generator output.")
    # Among those that forget properly, the one that disturbs least.
    best = min(winners, key=lambda row: row["bystander_drift"])
    return (f"TUNING. gamma={best['gamma']} forgets task 3 to "
            f"{best['target_after']:.1f}% and moves task 0 by "
            f"{best['bystander_drift']:.1f} points. Take it to T0.3 and re-run "
            f"the full sequence before believing it: one unlearn with one "
            f"protected task is the easiest case there is.")


def check(gammas, **kwargs):
    """Validate without training: build each config and report what would run."""
    print("Checking feasibility. Nothing is trained.\n")
    for gamma in gammas:
        config = make_config(sequence=1, limit_requests=3, gamma=gamma,
                             backbone=kwargs["backbone"], epochs=kwargs["epochs"])
        # The paper's notation: L to learn, U to unlearn.
        letter = {"learn": "L", "forget": "U"}
        requests = " ".join(f"{letter[a]}{t}" for a, t in config.requests)
        print(f"  gamma={gamma:<8} requests: {requests}   beta={config.beta} "
              f"burn_in={config.burn_in} chunks={config.chunks}")

    root = Path(kwargs["root"])
    print(f"\n  dataset root      {root}")
    print(f"  present           {root.exists()}")
    print(f"  tasks built       0 and 3 only, not all twenty")
    print(f"  output            None, so no files and no checkpoint collision")
    print(f"\n  accept if a gamma puts task 3 at or below {TARGET_AT_MOST}% "
          f"and moves task 0 by less than {BYSTANDER_DRIFT} points")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gammas", type=float, nargs="+",
                        default=[0.1, 0.01, 0.001, 0.0001])
    parser.add_argument("--backbone", default="resnet50",
                        choices=("cnn", "resnet18", "resnet50"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=None,
                        help="cap images per task; leave out for the real probe")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "gamma_probe.json")
    parser.add_argument("--quiet", action="store_true",
                        help="hide each run's own output")
    parser.add_argument("--check", action="store_true",
                        help="validate the setup and exit without training")
    args = parser.parse_args()

    settings = dict(backbone=args.backbone, epochs=args.epochs, root=str(args.root),
                    download=args.download, max_images=args.max_images,
                    verbose=not args.quiet)
    if args.check:
        return check(args.gammas, **settings)

    rows = []
    for gamma in args.gammas:
        print(f"\n{'=' * 64}\ngamma = {gamma}\n{'=' * 64}", flush=True)
        rows.append(probe_one(gamma, **settings))
        print(f"  task 3 {rows[-1]['target_after']:.1f}%, "
              f"task 0 {rows[-1]['bystander_before']:.1f}% -> "
              f"{rows[-1]['bystander_after']:.1f}%", flush=True)

    print("\n\n" + table(rows) + "\n")
    print(verdict(rows))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "task": "T0.1 gamma probe",
            "accept": {"target_at_most": TARGET_AT_MOST,
                       "bystander_drift_under": BYSTANDER_DRIFT},
            "backbone": args.backbone, "epochs": args.epochs,
            "rows": rows, "verdict": verdict(rows),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
