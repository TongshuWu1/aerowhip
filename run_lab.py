"""Open the guided cable-whip lab application. No job starts automatically."""
from pathlib import Path
import os


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    from deployment.lab_gui import main as launch
    return launch(root)


if __name__ == '__main__':
    raise SystemExit(main())
