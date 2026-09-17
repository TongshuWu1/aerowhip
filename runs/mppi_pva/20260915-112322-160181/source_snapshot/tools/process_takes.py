"""Process every immutable raw logger/Motive pair into versioned take artifacts."""

from __future__ import annotations

# Support both `python tools/<script>.py` and `python -m tools.<script>`.
if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from experimental_data.processing import process_all


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--take", help="Process only one take directory name.")
    parser.add_argument("--force", action="store_true", help="Regenerate unchanged outputs.")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()
    results = process_all(
        take_id=arguments.take,
        force=arguments.force,
        progress=print if arguments.verbose else None,
    )
    columns = ("Take", "Processed", "Frames", "Duration", "Sync RMS", "Command", "Fit Ready", "Status", "Warnings")
    print(" | ".join(columns))
    for result in results:
        print(
            " | ".join(
                (
                    str(result.get("take_id", "")),
                    "yes" if result.get("processed") else ("unchanged" if result.get("skipped_unchanged") else "no"),
                    str(result.get("frames", "-")),
                    f"{float(result['duration_s']):.3f}s" if "duration_s" in result else "-",
                    f"{1000.0*float(result['sync_rms_s']):.3f}ms" if "sync_rms_s" in result else "-",
                    f"{100.0*float(result['command_coverage']):.1f}%" if "command_coverage" in result else "-",
                    "yes" if result.get("fit_ready") else "no",
                    str(result.get("quality_status", "")),
                    "; ".join(str(value) for value in result.get("warnings", [])),
                )
            )
        )
    return 1 if any(result.get("quality_status") == "PROCESSING_FAILED" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
