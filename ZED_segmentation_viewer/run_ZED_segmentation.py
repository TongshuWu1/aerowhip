"""Launch the asynchronous ZED PIDNet segmentation viewer."""

import os
from pathlib import Path
import sys


os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
SOURCE_DIR = Path(__file__).resolve().parent / "source"
sys.path.insert(0, str(SOURCE_DIR))

from app import main


if __name__ == "__main__":
    main()
