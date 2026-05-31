"""CLI entry point: train one LSTM per ticker listed in config.yaml.

Run from the project root:
    python scripts/run_train.py
    python scripts/run_train.py --config /path/to/other_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file directly without an editable install.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.train import main  # noqa: E402  (sys.path tweak must come first)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a config.yaml. Defaults to the project root config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
