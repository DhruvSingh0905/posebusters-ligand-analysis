#!/usr/bin/env python3
"""Regenerate every finding in this repository. See `python run.py --list`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pb.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
