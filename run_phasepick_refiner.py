#!/usr/bin/env python
"""Small source-tree entry point for users who do not install the package."""

from pathlib import Path
import sys


repository_directory = Path(__file__).resolve().parent
source_directory = repository_directory / "src"
if str(source_directory) not in sys.path:
    sys.path.insert(0, str(source_directory))

from phasepick_refiner.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
