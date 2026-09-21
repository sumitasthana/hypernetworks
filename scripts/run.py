"""Run learn-and-unlearn sequences from the command line.

    python scripts/run.py --sequence 1 2 3 --seed 0 1 2 --backbone resnet50

A wrapper, so the work stays importable in `uncle.experiments` and nobody has
to remember `python -m`.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uncle.experiments import main

if __name__ == "__main__":
    main()
