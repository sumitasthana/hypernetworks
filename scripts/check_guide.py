"""Execute every code block in the Colab guide, with the heavy settings clamped.

Every failure this project has shipped to a notebook was code in the guide that
had been reasoned about and never run: a missing importlib.invalidate_caches,
a dataset path that stalled, a tuple unpack of a dict, a filename that only
existed after a particular run. They all reproduce in plain Python in seconds.

    python scripts/check_guide.py

Needs the dataset in data/tiny-imagenet-200. Runs on a CPU in a few minutes,
because every call is clamped to the small backbone, a few requests and a
hundred images per task. It checks that the code runs, not that the numbers
mean anything.
"""

import html
import json
import re
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uncle.baseline as baseline
import uncle.experiments as experiments

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "colab_guide.html"


def guide_blocks():
    """Every <pre><code> block in the guide, in order."""
    text = GUIDE.read_text(encoding="utf-8")
    return [html.unescape(block) for block in
            re.findall(r"<pre><code>(.*?)</code></pre>", text, flags=re.S)]


blocks = guide_blocks()

#: Blocks that are illustrations of output, not code to run.
SAMPLES = {4, 11, 12}
#: Blocks that only make sense inside Colab: git clone, drive.mount, GPU name.
COLAB_ONLY = {0, 1, 2}

SMALL = dict(backbone="cnn", chunks=8, epochs=1, burn_in=5, burn_in_min=5,
             limit_requests=3, max_images=100, progress=False, verbose=False)

_run = experiments.run_experiment
_seqs = experiments.run_sequences
_base = baseline.run_baseline


def small_run(*args, **kw):
    """Clamp a call to something that finishes, without changing its shape."""
    if kw.get("config") is not None:
        config = kw["config"]
        kw["config"] = replace(config, backbone="cnn", chunks=8, epochs=1,
                               burn_in=5, burn_in_min=5,
                               requests=config.requests[:3])
        for key in ("max_images", "progress", "verbose"):
            kw[key] = SMALL[key]
    else:
        kw.update(SMALL)
    return _run(*args, **kw)


def small_sequences(*args, **kw):
    kw.update(SMALL)
    kw["sequences"] = (1,)
    kw["seeds"] = (0,)
    return _seqs(*args, **kw)


def small_baseline(*args, **kw):
    kw.update(max_images=100, backbone="cnn", epochs=1, verbose=False)
    return _base(*args, **kw)


experiments.run_experiment = small_run
experiments.run_sequences = small_sequences
baseline.run_baseline = small_baseline

import tempfile


def main():
    save = Path(tempfile.mkdtemp(prefix="uncle-guide-"))
    env = {
        "DATA": str((ROOT / "data" / "tiny-imagenet-200").resolve()),
        "SAVE": str(save),
        "__name__": "__main__",
    }

    ok, failed = [], []
    for i, code in enumerate(blocks):
        if i in SAMPLES or i in COLAB_ONLY:
            continue
        try:
            exec(compile(code, f"<block {i}>", "exec"), env)
            ok.append(i)
            print(f"[{i:>2}] ok", flush=True)
        except Exception as error:
            failed.append((i, f"{type(error).__name__}: {error}"))
            print(f"[{i:>2}] FAILED  {type(error).__name__}: {error}", flush=True)
            traceback.print_exc(limit=2)

    print(f"\nran {len(ok)}   failed {len(failed)}")
    for i, why in failed:
        print(f"  [{i}] {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
