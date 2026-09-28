"""
check_grid_spacing.py

Simple diagnostic for checking the physical grid spacing used for
32x32, 64x64, 128x128, etc.

This does NOT modify training.
It is only a sanity check.

For the SWE bathymetry problem:

    x in [-2.5, 2.5]
    y in [-2.5, 2.5]

so

    Lx = 5.0
    Ly = 5.0

and the code convention currently used by the training script is

    dx = Lx / nx
    dy = Ly / ny

Examples
--------
64x64:

    python scripts/check_grid_spacing.py --resolution 64

128x128:

    python scripts/check_grid_spacing.py --resolution 128

32x32:

    python scripts/check_grid_spacing.py --resolution 32
"""

from __future__ import annotations

import argparse


# =============================================================================
# ARGUMENTS
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check SWE spatial grid spacing."
    )

    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        help="Spatial resolution, e.g. 32, 64, or 128.",
    )

    parser.add_argument(
        "--x-length",
        type=float,
        default=5.0,
        help="Physical domain length in x.",
    )

    parser.add_argument(
        "--y-length",
        type=float,
        default=5.0,
        help="Physical domain length in y.",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================


def main():
    args = parse_args()

    nx = int(args.resolution)
    ny = int(args.resolution)

    if nx <= 0 or ny <= 0:
        raise ValueError("Resolution must be positive.")

    dx = float(args.x_length) / nx
    dy = float(args.y_length) / ny

    print()
    print("=" * 70)
    print("SWE GRID-SPACING DIAGNOSTIC")
    print("=" * 70)

    print(f"Resolution : {nx} x {ny}")
    print(f"Lx         : {args.x_length:.10f}")
    print(f"Ly         : {args.y_length:.10f}")
    print(f"dx         : {dx:.10f}")
    print(f"dy         : {dy:.10f}")

    print("-" * 70)

    expected = {
        32: 0.15625,
        64: 0.078125,
        128: 0.0390625,
    }

    if nx in expected:
        expected_dx = expected[nx]

        print(f"Expected dx for {nx}x{nx}: {expected_dx:.10f}")

        error = abs(dx - expected_dx)

        if error < 1.0e-12:
            print("dx check   : PASS")
        else:
            print("dx check   : FAIL")

    if ny in expected:
        expected_dy = expected[ny]

        print(f"Expected dy for {ny}x{ny}: {expected_dy:.10f}")

        error = abs(dy - expected_dy)

        if error < 1.0e-12:
            print("dy check   : PASS")
        else:
            print("dy check   : FAIL")

    print("=" * 70)
    print()


if __name__ == "__main__":
    main()