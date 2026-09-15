"""
Generate 2D shallow-water trajectories over varying bathymetry.

Run this file in the `swegen` conda environment.

Modes
-----
Lake-at-rest verification:
    python scripts/generate_swe_bathy.py --mode lake

ID dataset:
    python scripts/generate_swe_bathy.py --mode id --train-count 24 --eval-count 6 --test-count 6

OOD dataset:
    python scripts/generate_swe_bathy.py --mode ood --num-ood 6

State
-----
q[..., 0] = h
q[..., 1] = hu
q[..., 2] = hv

Auxiliary field
---------------
b(x,y) = bathymetry

The free-surface elevation is

    eta = h + b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = PROJECT_ROOT / "utils"

# Import bathymetry.py directly so that running in `swegen`
# does not execute utils/__init__.py.
#
# utils/__init__.py imports checkpoint / Orbax code, which
# belongs to the `cfo` environment and is not required here.
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from bathymetry import (
    TerrainParameters,
    make_bathymetry,
    parameter_vector,
    sample_id_parameters,
    sample_ood_parameters,
)


# ================================================================
# ARGUMENTS
# ================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Generate variable-bottom 2D SWE trajectories with PyClaw."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "lake",
            "id",
            "ood",
        ],
        default="lake",
    )

    parser.add_argument(
        "--nx",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--ny",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--x-min",
        type=float,
        default=-2.5,
    )

    parser.add_argument(
        "--x-max",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--y-min",
        type=float,
        default=-2.5,
    )

    parser.add_argument(
        "--y-max",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--num-times",
        type=int,
        default=51,
    )

    parser.add_argument(
        "--tfinal",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--g",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--dry-tolerance",
        type=float,
        default=1.0e-6,
    )

    parser.add_argument(
        "--sea-level",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--eta-inside",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--eta-outside",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lake-eta",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--dam-radius-min",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--dam-radius-max",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.20,
        help=(
            "Abort if the initial water depth becomes "
            "<= this value."
        ),
    )

    parser.add_argument(
        "--train-count",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--eval-count",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--test-count",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--num-ood",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit HDF5 output path.",
    )

    return parser.parse_args()


# ================================================================
# GRID
# ================================================================

def build_grid(
    nx: int,
    ny: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
):

    dx = (
        x_max
        -
        x_min
    ) / nx

    dy = (
        y_max
        -
        y_min
    ) / ny

    x = (
        x_min
        +
        (
            np.arange(nx)
            +
            0.5
        )
        *
        dx
    )

    y = (
        y_min
        +
        (
            np.arange(ny)
            +
            0.5
        )
        *
        dy
    )

    X, Y = np.meshgrid(
        x,
        y,
        indexing="ij",
    )

    return (
        x,
        y,
        X,
        Y,
    )


# ================================================================
# RIEMANN SOLVER PARAMETER SETUP
# ================================================================

def _required_cparam_names(
    riemann_solver,
) -> list[str]:

    """
    Return the names inside the Fortran cparam common block used by
    the installed Riemann solver.

    PyClaw requires every one of these names to be present in
    state.problem_data.
    """

    if not hasattr(
        riemann_solver,
        "cparam",
    ):
        return []

    names = [
        name
        for name
        in dir(
            riemann_solver.cparam
        )
        if not name.startswith("_")
    ]

    # Remove duplicates while keeping original order.
    names = list(
        dict.fromkeys(
            names
        )
    )

    return names


def configure_riemann_problem_data(
    *,
    state,
    riemann_solver,
    gravity: float,
    dry_tolerance: float,
    sea_level: float,
) -> None:

    """
    Populate state.problem_data and verify that PyClaw can transfer
    those parameters into the Fortran Riemann solver.

    This is intentionally performed BEFORE claw.run(), so a missing
    parameter produces a useful error immediately.
    """

    # ------------------------------------------------------------
    # Standard Clawpack shallow-water f-wave parameter names
    # ------------------------------------------------------------

    state.problem_data[
        "grav"
    ] = float(
        gravity
    )

    state.problem_data[
        "dry_tolerance"
    ] = float(
        dry_tolerance
    )

    state.problem_data[
        "sea_level"
    ] = float(
        sea_level
    )

    # ------------------------------------------------------------
    # Ask this exact installed solver what its Fortran common block
    # requires.
    # ------------------------------------------------------------

    required = _required_cparam_names(
        riemann_solver
    )

    gravity_names = {
        "grav",
        "g",
        "gravity",
    }

    dry_names = {
        "dry_tolerance",
        "drytol",
        "dry_tol",
        "drytolerance",
    }

    sea_names = {
        "sea_level",
        "sealevel",
        "sea_level0",
    }

    unknown = []

    # ------------------------------------------------------------
    # Handle aliases if this installed Clawpack version uses slightly
    # different names.
    # ------------------------------------------------------------

    for name in required:

        normalized = (
            name
            .lower()
        )

        if normalized in gravity_names:

            state.problem_data[
                name
            ] = float(
                gravity
            )

        elif normalized in dry_names:

            state.problem_data[
                name
            ] = float(
                dry_tolerance
            )

        elif normalized in sea_names:

            state.problem_data[
                name
            ] = float(
                sea_level
            )

        elif name not in (
            state.problem_data
        ):

            unknown.append(
                name
            )

    # ------------------------------------------------------------
    # Print exactly what your Clawpack build expects.
    # ------------------------------------------------------------

    print(
        "\n"
        +
        "=" * 60
    )

    print(
        "RIEMANN SOLVER PARAMETER CHECK"
    )

    print(
        "=" * 60
    )

    print(
        "\nRequired cparam names from this "
        "Clawpack installation:"
    )

    if required:

        for name in required:

            print(
                f"  {name}"
            )

    else:

        print(
            "  No cparam common block exposed."
        )

    print(
        "\nstate.problem_data:"
    )

    for (
        key,
        value,
    ) in (
        state.problem_data.items()
    ):

        print(
            f"  {key} = {value}"
        )

    # ------------------------------------------------------------
    # Do not guess unknown physical parameters.
    # ------------------------------------------------------------

    if unknown:

        raise RuntimeError(
            "\nThe installed Riemann solver requires "
            "cparam name(s) that this script does not "
            "yet know how to map:\n\n  "
            +
            "\n  ".join(
                unknown
            )
            +
            "\n\nCopy those names exactly from the terminal output."
        )

    # ------------------------------------------------------------
    # Test transfer into Fortran common block NOW.
    # ------------------------------------------------------------

    try:

        state.set_cparam(
            riemann_solver
        )

    except Exception as exc:

        raise RuntimeError(
            "\nPyClaw could not copy problem_data "
            "into the Fortran cparam common block.\n"
            f"Required names: {required}\n"
            f"Provided keys: "
            f"{list(state.problem_data.keys())}\n"
        ) from exc

    print(
        "\nRiemann cparam configuration: OK"
    )

    print(
        "=" * 60
    )


# ================================================================
# ONE PYCLAW SIMULATION
# ================================================================

def run_pyclaw(
    *,
    bathymetry: np.ndarray,
    dam_radius: float,
    num_times: int,
    tfinal: float,
    g: float,
    dry_tolerance: float,
    sea_level: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    eta_inside: float,
    eta_outside: float,
    min_depth: float,
    lake_at_rest: bool = False,
    lake_eta: float = 2.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    try:

        from clawpack import (
            pyclaw,
            riemann,
        )

    except ImportError as exc:

        raise RuntimeError(
            "Clawpack/PyClaw is not available. "
            "Activate the `swegen` environment."
        ) from exc

    # ------------------------------------------------------------
    # Confirm bathymetry solver exists
    # ------------------------------------------------------------

    if not hasattr(
        riemann,
        "shallow_bathymetry_fwave_2D",
    ):

        raise RuntimeError(
            "This Clawpack installation does not expose "
            "`riemann.shallow_bathymetry_fwave_2D`."
        )

    rp = (
        riemann
        .shallow_bathymetry_fwave_2D
    )

    bathymetry = np.asarray(
        bathymetry,
        dtype=np.float64,
    )

    if bathymetry.ndim != 2:

        raise ValueError(
            "bathymetry must have shape "
            f"(nx, ny); got {bathymetry.shape}"
        )

    if not np.all(
        np.isfinite(
            bathymetry
        )
    ):

        raise ValueError(
            "Bathymetry contains NaN or Inf."
        )

    (
        nx,
        ny,
    ) = (
        bathymetry.shape
    )

    # ------------------------------------------------------------
    # SOLVER
    # ------------------------------------------------------------

    solver = pyclaw.ClawSolver2D(
        rp
    )

    # Bathymetry Riemann solver uses f-waves.
    solver.fwave = True

    # TVD limiter.
    solver.limiters = (
        pyclaw
        .limiters
        .tvd
        .MC
    )

    # Use dimensional splitting.
    #
    # This avoids requiring a separate bathymetry-aware transverse
    # Riemann solver.
    solver.dimensional_split = True

    # ------------------------------------------------------------
    # CONSERVED-STATE BOUNDARY CONDITIONS
    # ------------------------------------------------------------

    solver.bc_lower[
        0
    ] = pyclaw.BC.extrap

    solver.bc_upper[
        0
    ] = pyclaw.BC.extrap

    solver.bc_lower[
        1
    ] = pyclaw.BC.extrap

    solver.bc_upper[
        1
    ] = pyclaw.BC.extrap

    # ------------------------------------------------------------
    # BATHYMETRY AUXILIARY-FIELD BOUNDARY CONDITIONS
    # ------------------------------------------------------------

    solver.aux_bc_lower[
        0
    ] = pyclaw.BC.extrap

    solver.aux_bc_upper[
        0
    ] = pyclaw.BC.extrap

    solver.aux_bc_lower[
        1
    ] = pyclaw.BC.extrap

    solver.aux_bc_upper[
        1
    ] = pyclaw.BC.extrap

    # ------------------------------------------------------------
    # DOMAIN
    # ------------------------------------------------------------

    xdim = pyclaw.Dimension(
        x_min,
        x_max,
        nx,
        name="x",
    )

    ydim = pyclaw.Dimension(
        y_min,
        y_max,
        ny,
        name="y",
    )

    domain = pyclaw.Domain(
        [
            xdim,
            ydim,
        ]
    )

    # ------------------------------------------------------------
    # STATE
    #
    # q[0] = h
    # q[1] = hu
    # q[2] = hv
    #
    # aux[0] = bathymetry b
    # ------------------------------------------------------------

    state = pyclaw.State(
        domain,
        3,
        1,
    )

    # ------------------------------------------------------------
    # IMPORTANT:
    # Configure Fortran Riemann-solver parameters BEFORE run().
    # ------------------------------------------------------------

    configure_riemann_problem_data(
        state=state,
        riemann_solver=rp,
        gravity=g,
        dry_tolerance=dry_tolerance,
        sea_level=sea_level,
    )

    # ------------------------------------------------------------
    # BATHYMETRY
    # ------------------------------------------------------------

    state.aux[
        0,
        :,
        :,
    ] = bathymetry

    # ------------------------------------------------------------
    # CELL-CENTER COORDINATES
    # ------------------------------------------------------------

    (
        X,
        Y,
    ) = (
        state.p_centers
    )

    # ------------------------------------------------------------
    # INITIAL FREE-SURFACE ELEVATION ETA
    # ------------------------------------------------------------

    if lake_at_rest:

        eta = np.full_like(
            X,
            float(
                lake_eta
            ),
            dtype=np.float64,
        )

    else:

        radius = np.sqrt(
            X**2
            +
            Y**2
        )

        eta = np.where(
            radius
            <=
            float(
                dam_radius
            ),
            float(
                eta_inside
            ),
            float(
                eta_outside
            ),
        )

    # ------------------------------------------------------------
    # ETA = H + B
    #
    # therefore
    #
    # H = ETA - B
    # ------------------------------------------------------------

    h = (
        eta
        -
        bathymetry
    )

    if not np.all(
        np.isfinite(
            h
        )
    ):

        raise ValueError(
            "Initial depth h contains NaN or Inf."
        )

    h_min = float(
        np.min(
            h
        )
    )

    if h_min <= float(
        min_depth
    ):

        raise ValueError(
            "\nInitial depth is too small.\n"
            f"minimum h = {h_min:.8e}\n"
            f"required h > {min_depth:.8e}\n"
            "Reduce terrain amplitude or increase eta."
        )

    # ------------------------------------------------------------
    # INITIAL CONSERVED VARIABLES
    # ------------------------------------------------------------

    state.q[
        0,
        :,
        :,
    ] = h

    state.q[
        1,
        :,
        :,
    ] = 0.0

    state.q[
        2,
        :,
        :,
    ] = 0.0

    # ------------------------------------------------------------
    # SOLUTION
    # ------------------------------------------------------------

    solution = pyclaw.Solution(
        state,
        domain,
    )

    # ------------------------------------------------------------
    # CONTROLLER
    # ------------------------------------------------------------

    claw = pyclaw.Controller()

    claw.solver = solver

    claw.solution = solution

    claw.tfinal = float(
        tfinal
    )

    # num_output_times excludes t=0.
    #
    # For 51 frames:
    #
    # initial frame + 50 outputs = 51 total.
    claw.num_output_times = int(
        num_times
        -
        1
    )

    claw.keep_copy = True

    # Do not write normal PyClaw frame files to disk.
    # We keep the frames directly in memory.
    claw.output_format = None

    # ------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------

    print(
        "\nStarting PyClaw simulation..."
    )

    status = claw.run()

    print(
        "PyClaw simulation finished."
    )

    if status is not None:

        print(
            f"PyClaw status: {status}"
        )

    # ------------------------------------------------------------
    # COLLECT FRAMES
    # ------------------------------------------------------------

    frames = (
        claw.frames
    )

    if len(
        frames
    ) != num_times:

        raise RuntimeError(
            f"Expected {num_times} frames "
            f"but received {len(frames)}."
        )

    q_list = []

    times = []

    for frame in frames:

        q_frame = np.moveaxis(
            np.asarray(
                frame.state.q,
                dtype=np.float64,
            ),
            0,
            -1,
        )

        q_list.append(
            q_frame.astype(
                np.float32
            )
        )

        times.append(
            float(
                frame.state.t
            )
        )

    q = np.stack(
        q_list,
        axis=0,
    )

    times = np.asarray(
        times,
        dtype=np.float32,
    )

    # ------------------------------------------------------------
    # FINAL SAFETY CHECKS
    # ------------------------------------------------------------

    expected_shape = (
        num_times,
        nx,
        ny,
        3,
    )

    if q.shape != expected_shape:

        raise RuntimeError(
            "Unexpected trajectory shape.\n"
            f"Expected: {expected_shape}\n"
            f"Received: {q.shape}"
        )

    if not np.all(
        np.isfinite(
            q
        )
    ):

        raise RuntimeError(
            "Simulation produced NaN or Inf."
        )

    minimum_depth = float(
        np.min(
            q[
                ...,
                0,
            ]
        )
    )

    if minimum_depth <= 0.0:

        raise RuntimeError(
            "Simulation produced non-positive depth: "
            f"{minimum_depth:.8e}"
        )

    return (
        q,
        times,
    )


# ================================================================
# WRITE ONE DATASET SPLIT
# ================================================================

def write_split(
    h5: h5py.File,
    split_name: str,
    *,
    params_list: list[
        TerrainParameters
    ],
    rng: np.random.Generator,
    X: np.ndarray,
    Y: np.ndarray,
    args: argparse.Namespace,
) -> None:

    q_all = []

    b_all = []

    param_all = []

    radius_all = []

    times_ref = None

    print(
        "\n"
        +
        "=" * 60
    )

    print(
        f"GENERATING SPLIT: "
        f"{split_name}"
    )

    print(
        f"NUMBER OF TRAJECTORIES: "
        f"{len(params_list)}"
    )

    print(
        "=" * 60
    )

    for (
        index,
        params,
    ) in enumerate(
        params_list
    ):

        b = make_bathymetry(
            X,
            Y,
            params,
        ).astype(
            np.float32
        )

        dam_radius = float(
            rng.uniform(
                args.dam_radius_min,
                args.dam_radius_max,
            )
        )

        print(
            f"\nTrajectory "
            f"{index + 1}/"
            f"{len(params_list)}"
        )

        print(
            f"  kind       = "
            f"{params.kind}"
        )

        print(
            f"  b_max      = "
            f"{float(np.max(b)):.6f}"
        )

        print(
            f"  dam radius = "
            f"{dam_radius:.4f}"
        )

        q, times = run_pyclaw(
            bathymetry=b,
            dam_radius=dam_radius,
            num_times=args.num_times,
            tfinal=args.tfinal,
            g=args.g,
            dry_tolerance=args.dry_tolerance,
            sea_level=args.sea_level,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            eta_inside=args.eta_inside,
            eta_outside=args.eta_outside,
            min_depth=args.min_depth,
            lake_at_rest=False,
            lake_eta=args.lake_eta,
        )

        if times_ref is None:

            times_ref = (
                times.copy()
            )

        elif not np.allclose(
            times_ref,
            times,
        ):

            raise RuntimeError(
                "Stored output times changed "
                "between trajectories."
            )

        q_all.append(
            q
        )

        b_all.append(
            b[
                ...,
                None,
            ]
        )

        param_all.append(
            parameter_vector(
                params
            )
        )

        radius_all.append(
            dam_radius
        )

        print(
            f"  h_min      = "
            f"{float(np.min(q[..., 0])):.6f}"
        )

        print(
            f"  h_max      = "
            f"{float(np.max(q[..., 0])):.6f}"
        )

    group = h5.create_group(
        split_name
    )

    group.create_dataset(
        "q",
        data=np.asarray(
            q_all,
            dtype=np.float32,
        ),
        compression="gzip",
    )

    group.create_dataset(
        "bathymetry",
        data=np.asarray(
            b_all,
            dtype=np.float32,
        ),
        compression="gzip",
    )

    group.create_dataset(
        "terrain_parameters",
        data=np.asarray(
            param_all,
            dtype=np.float32,
        ),
    )

    group.create_dataset(
        "dam_radius",
        data=np.asarray(
            radius_all,
            dtype=np.float32,
        ),
    )

    group.create_dataset(
        "time",
        data=np.asarray(
            times_ref,
            dtype=np.float32,
        ),
    )


# ================================================================
# MAIN
# ================================================================

def main() -> None:

    args = parse_args()

    # ------------------------------------------------------------
    # BUILD GRID
    # ------------------------------------------------------------

    (
        x,
        y,
        X,
        Y,
    ) = build_grid(
        args.nx,
        args.ny,
        args.x_min,
        args.x_max,
        args.y_min,
        args.y_max,
    )

    dx = (
        args.x_max
        -
        args.x_min
    ) / args.nx

    dy = (
        args.y_max
        -
        args.y_min
    ) / args.ny

    print(
        "=" * 60
    )

    print(
        "2D SWE BATHYMETRY GENERATOR"
    )

    print(
        "=" * 60
    )

    print(
        f"mode          = "
        f"{args.mode}"
    )

    print(
        f"grid          = "
        f"{args.nx} x {args.ny}"
    )

    print(
        f"dx            = "
        f"{dx:.8f}"
    )

    print(
        f"dy            = "
        f"{dy:.8f}"
    )

    print(
        f"g             = "
        f"{args.g}"
    )

    print(
        f"dry_tolerance = "
        f"{args.dry_tolerance}"
    )

    print(
        f"sea_level     = "
        f"{args.sea_level}"
    )

    # ------------------------------------------------------------
    # OUTPUT DIRECTORY
    # ------------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        /
        "data"
        /
        "shallow_water_bathy"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # OUTPUT FILE
    # ------------------------------------------------------------

    if args.output is not None:

        output_path = (
            Path(
                args.output
            )
            .expanduser()
            .resolve()
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    elif args.mode == "lake":

        output_path = (
            output_dir
            /
            f"swe_bathy_{args.nx}_lake_test.h5"
        )

    elif args.mode == "id":

        output_path = (
            output_dir
            /
            f"swe_bathy_{args.nx}_id.h5"
        )

    else:

        output_path = (
            output_dir
            /
            f"swe_bathy_{args.nx}_ood.h5"
        )

    # ------------------------------------------------------------
    # RNG
    # ------------------------------------------------------------

    rng = np.random.default_rng(
        args.seed
    )

    # ============================================================
    # PHASE 46
    # LAKE-AT-REST TEST
    # ============================================================

    if args.mode == "lake":

        params = (
            sample_id_parameters(
                rng
            )
        )

        b = make_bathymetry(
            X,
            Y,
            params,
        ).astype(
            np.float32
        )

        print(
            "\nLake-at-rest terrain:"
        )

        print(
            f"  kind  = "
            f"{params.kind}"
        )

        print(
            f"  b_min = "
            f"{float(np.min(b)):.8f}"
        )

        print(
            f"  b_max = "
            f"{float(np.max(b)):.8f}"
        )

        q, times = run_pyclaw(
            bathymetry=b,
            dam_radius=0.5,
            num_times=args.num_times,
            tfinal=args.tfinal,
            g=args.g,
            dry_tolerance=args.dry_tolerance,
            sea_level=args.sea_level,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            eta_inside=args.eta_inside,
            eta_outside=args.eta_outside,
            min_depth=args.min_depth,
            lake_at_rest=True,
            lake_eta=args.lake_eta,
        )

        # --------------------------------------------------------
        # FREE-SURFACE HEIGHT
        #
        # eta = h + b
        # --------------------------------------------------------

        eta = (
            q[
                ...,
                0,
            ]
            +
            b[
                None,
                :,
                :,
            ]
        )

        eta_error = float(
            np.max(
                np.abs(
                    eta
                    -
                    float(
                        args.lake_eta
                    )
                )
            )
        )

        momentum_error = float(
            np.max(
                np.abs(
                    q[
                        ...,
                        1:3,
                    ]
                )
            )
        )

        # --------------------------------------------------------
        # SAVE LAKE TEST
        # --------------------------------------------------------

        with h5py.File(
            output_path,
            "w",
        ) as h5:

            h5.create_dataset(
                "q",
                data=q,
                compression="gzip",
            )

            h5.create_dataset(
                "bathymetry",
                data=b[
                    ...,
                    None,
                ],
                compression="gzip",
            )

            h5.create_dataset(
                "time",
                data=times,
            )

            h5.create_dataset(
                "x",
                data=x.astype(
                    np.float32
                ),
            )

            h5.create_dataset(
                "y",
                data=y.astype(
                    np.float32
                ),
            )

            h5.attrs[
                "lake_eta"
            ] = float(
                args.lake_eta
            )

            h5.attrs[
                "gravity"
            ] = float(
                args.g
            )

            h5.attrs[
                "dry_tolerance"
            ] = float(
                args.dry_tolerance
            )

            h5.attrs[
                "sea_level"
            ] = float(
                args.sea_level
            )

        # --------------------------------------------------------
        # REPORT
        # --------------------------------------------------------

        print(
            "\n"
            +
            "=" * 60
        )

        print(
            "LAKE-AT-REST SOLVER CHECK"
        )

        print(
            "=" * 60
        )

        print(
            f"max |eta - eta0| = "
            f"{eta_error:.8e}"
        )

        print(
            f"max |hu, hv|     = "
            f"{momentum_error:.8e}"
        )

        print(
            f"minimum h        = "
            f"{float(np.min(q[..., 0])):.8e}"
        )

        print(
            f"maximum h        = "
            f"{float(np.max(q[..., 0])):.8e}"
        )

        print(
            f"saved            = "
            f"{output_path}"
        )

        print(
            "=" * 60
        )

        return

    # ============================================================
    # PHASE 47:
    # ID DATASET
    #
    # PHASE 56:
    # OOD DATASET
    # ============================================================

    if output_path.exists():

        print(
            f"\nRemoving existing output file: "
            f"{output_path}"
        )

        output_path.unlink()

    with h5py.File(
        output_path,
        "w",
    ) as h5:

        # --------------------------------------------------------
        # GLOBAL METADATA
        # --------------------------------------------------------

        h5.attrs[
            "nx"
        ] = int(
            args.nx
        )

        h5.attrs[
            "ny"
        ] = int(
            args.ny
        )

        h5.attrs[
            "x_min"
        ] = float(
            args.x_min
        )

        h5.attrs[
            "x_max"
        ] = float(
            args.x_max
        )

        h5.attrs[
            "y_min"
        ] = float(
            args.y_min
        )

        h5.attrs[
            "y_max"
        ] = float(
            args.y_max
        )

        h5.attrs[
            "gravity"
        ] = float(
            args.g
        )

        h5.attrs[
            "dry_tolerance"
        ] = float(
            args.dry_tolerance
        )

        h5.attrs[
            "sea_level"
        ] = float(
            args.sea_level
        )

        h5.attrs[
            "eta_inside"
        ] = float(
            args.eta_inside
        )

        h5.attrs[
            "eta_outside"
        ] = float(
            args.eta_outside
        )

        h5.create_dataset(
            "x",
            data=x.astype(
                np.float32
            ),
        )

        h5.create_dataset(
            "y",
            data=y.astype(
                np.float32
            ),
        )

        # --------------------------------------------------------
        # ID DATA
        # --------------------------------------------------------

        if args.mode == "id":

            split_counts = {
                "train":
                    args.train_count,

                "eval":
                    args.eval_count,

                "test_id":
                    args.test_count,
            }

            for (
                split_name,
                count,
            ) in (
                split_counts.items()
            ):

                params_list = [
                    sample_id_parameters(
                        rng
                    )
                    for _
                    in range(
                        count
                    )
                ]

                write_split(
                    h5,
                    split_name,
                    params_list=params_list,
                    rng=rng,
                    X=X,
                    Y=Y,
                    args=args,
                )

        # --------------------------------------------------------
        # OOD DATA
        # --------------------------------------------------------

        elif args.mode == "ood":

            params_list = [
                sample_ood_parameters(
                    rng,
                    index,
                )
                for index
                in range(
                    args.num_ood
                )
            ]

            write_split(
                h5,
                "test_ood",
                params_list=params_list,
                rng=rng,
                X=X,
                Y=Y,
                args=args,
            )

    # ------------------------------------------------------------
    # DONE
    # ------------------------------------------------------------

    print(
        "\n"
        +
        "=" * 60
    )

    print(
        "DATASET GENERATION COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"saved = "
        f"{output_path}"
    )


if __name__ == "__main__":

    main()