"""Build motion-state and estimated-force data for the simplified model."""

from __future__ import annotations

# Support both `python tools/<script>.py` and `python -m tools.<script>`.
if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from experimental_data.force_dataset import build_force_dataset, format_build_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--take", help="Build only one take.")
    parser.add_argument(
        "--include-untouched-test",
        action="store_true",
        help="Explicitly allow conversion of the protected untouched test take.",
    )
    arguments = parser.parse_args()
    manifest = build_force_dataset(
        take_id=arguments.take,
        include_untouched_test=arguments.include_untouched_test,
    )
    for row in format_build_rows(manifest):
        print(row)
    summary = manifest["summary"]
    print(
        f"Built {summary['take_count']} takes, {float(summary['duration_s']):.2f} s; "
        "force is estimated, not measured."
    )
    if manifest["excluded_takes"]:
        print(
            "Excluded: "
            + ", ".join(
                f"{name} ({reason})"
                for name, reason in manifest["excluded_takes"].items()
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
