"""Train one task with ordinary backprop, as a sanity check.

    python scripts/baseline.py --task 1 --backbone resnet18 --epochs 10

No hypernetwork. If this cannot beat 10% on ten classes, nothing downstream is
worth debugging.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uncle.baseline import main

if __name__ == "__main__":
    main()
