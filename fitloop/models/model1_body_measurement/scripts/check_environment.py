#!/usr/bin/env python3
"""
scripts/check_environment.py
-----------------------------
Verify that all Model 1 dependencies are importable and report their versions.

Run with:
    conda run -n ml_env python scripts/check_environment.py
"""

import importlib
import platform
import sys

REQUIRED = [
    ("cv2",       "opencv-python"),
    ("mediapipe", "mediapipe"),
    ("numpy",     "numpy"),
    ("PIL",       "Pillow"),
]

OPTIONAL = [
    ("matplotlib", "matplotlib"),
    ("pytest",     "pytest"),
]


def check(packages: list[tuple[str, str]], label: str) -> list[str]:
    failures = []
    print(f"\n{label}")
    print("-" * len(label))
    for mod, pip_name in packages:
        try:
            m = importlib.import_module(mod)
            version = getattr(m, "__version__", "unknown")
            print(f"  ✓  {pip_name:<22} {version}")
        except ImportError as exc:
            print(f"  ✗  {pip_name:<22} MISSING  ({exc})")
            failures.append(pip_name)
    return failures


def main() -> None:
    print("\n=== FitLoop Model 1 — Environment Check ===")
    print(f"Python      : {sys.version}")
    print(f"Interpreter : {sys.executable}")
    print(f"Platform    : {platform.platform()}")
    print(f"Machine     : {platform.machine()}")

    fails = check(REQUIRED, "Required packages")
    check(OPTIONAL, "Optional packages")

    print()
    if fails:
        print(f"✗ {len(fails)} required package(s) missing: {', '.join(fails)}")
        print("  Install with:  pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("✓ All required packages are available. Environment looks good.")


if __name__ == "__main__":
    main()
