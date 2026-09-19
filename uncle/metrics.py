"""The paper's four numbers, computed from a run's history.

Accuracy alone hides the two problems the paper is about, so it reports four:

  RA       average accuracy of tasks still retained at the end. Higher is better.
  FA       average accuracy of tasks forgotten. Should sit at chance.
  Spill    how much a forget request disturbed every other task. Lower is better.
  Relapse  how much a forgotten task crept back by the end. Lower is better.
"""


def spill(record: dict) -> float | None:
    """How far every other task moved during one forget request. Paper eq. 4."""
    if record["action"] != "forget":
        return None

    return sum(
        abs(record["after"][task] - record["before"][task])
        for task in record["before"]
        if task != record["task"]
    )


def relapse(history: list[dict]) -> dict[str, float]:
    """How much each forgotten task recovered by the end of the run. Paper eq. 5."""
    if not history:
        return {}

    final = history[-1]["after"]
    recovered = {}

    for record in history:
        if record["action"] == "forget":
            task = record["task"]
            recovered[task] = abs(final[task] - record["after"][task])

    return recovered


def summary(history: list[dict]) -> dict:
    """The four headline numbers for one run."""
    if not history:
        return {}

    final = history[-1]
    forgotten = set(final["forgotten"])
    retained = [task for task in final["seen"] if task not in forgotten]

    spills = [spill(record) for record in history if record["action"] == "forget"]
    recovered = relapse(history)

    return {
        "retain_accuracy": _mean(final["after"][task] for task in retained),
        "forget_accuracy": _mean(final["after"][task] for task in forgotten),
        "mean_spill": _mean(spills),
        "mean_relapse": _mean(recovered.values()),
        "spill_per_request": spills,
        "relapse_per_task": recovered,
    }


def _mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None
