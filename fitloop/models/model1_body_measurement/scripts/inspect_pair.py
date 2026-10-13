#!/usr/bin/env python3
"""
scripts/inspect_pair.py
-----------------------
CLI tool to inspect and validate front + side paired captures.

Usage::

    conda run -n ml_env python scripts/inspect_pair.py \\
        --front data/raw/front.jpg \\
        --side data/raw/side.jpg \\
        --height 175
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitloop_measurement.quality import PairedQualityChecker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect paired front/side images for quality and calibration.")
    p.add_argument("--front", required=True, help="Path to front image")
    p.add_argument("--side", required=True, help="Path to side image")
    p.add_argument("--height", type=float, default=None, help="User known height in cm")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("\n=== FitLoop Model 1 — Paired Image Inspection ===")
    print(f"Front: {args.front}")
    print(f"Side : {args.side}")
    print(f"Height: {args.height} cm\n")

    with PairedQualityChecker() as checker:
        report = checker.validate_pair(
            front_image=args.front,
            side_image=args.side,
            subject_height_cm=args.height,
        )

        print(report)

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
