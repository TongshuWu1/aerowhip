"""Launch the PIDNet collection, annotation, and training application."""

import os
from pathlib import Path
import sys


os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
SOURCE_DIR = Path(__file__).resolve().parent / "source"
sys.path.insert(0, str(SOURCE_DIR))

from pidnet_training_gui import main


if __name__ == "__main__":
    main()
