"""Make the stage folders importable (their names start with digits, so they
cannot be packages) and force a headless matplotlib backend."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
for stage in sorted(ROOT.glob("0[1-7]-*")):
    sys.path.insert(0, str(stage))
