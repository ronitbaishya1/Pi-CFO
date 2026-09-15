"""
Controlled Gaussian-bathymetry counterfactual experiment.

Purpose
-------
Answer:

    Did the neural operator actually learn what the Gaussian hill
    does to the shallow-water flow?

We generate four cases with exactly the same initial free surface:

    1. flat bed
    2. Gaussian hill on the left
    3. Gaussian hill in the center
    4. Gaussian hill on the right

Only b(x,y) changes.

The physically meaningful initial condition is held fixed in terms of

    eta = h + b

so

    h = eta - b

and initially

    hu = hv = 0.

Output
------
data/shallow_water_bathy/
    gaussian_counterfactual_32.h5

Run in the swegen environment:

    conda activate swegen

    python scripts/generate_gaussian_counterfactual.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np

from clawpack import pyclaw
from clawpack import riemann


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ================================================================
# ARGUMENTS
# ================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Generate matched flat/Gaussian "
            "shallow-water counterfactual cases."
        )
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
        "--xlower",
        type=float,
        default=-2.5,
    )

    parser.add_argument(
        "--xupper",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--ylower",
        type=float,
        default=-2.5,
    )

    parser.add_argument(
        "--yupper",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--tfinal",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--num-snapshots",
        type=int,
        default=51,
    )

    parser.add_argument(
        "--gravity",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--dam-radius",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--hill-amplitude",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--hill-sigma",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--hill-offset",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "gaussian_counterfactual_32.h5"
        ),
    )

    return parser.parse_args()


# ================================================================
# GAUSSIAN BATHYMETRY
# ================================================================

def gaussian_bathymetry(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    amplitude: float,
    xc: float,
    yc: float,
    sigma: float,
) -> np.ndarray:

    radius_squared = (
        (X - xc) ** 2
        +
        (Y - yc) ** 2
    )

    return (
        amplitude
        *
        np.exp(
            -0.5
            *
            radius_squared
            /
            sigma**2
        )
    )


# ================================================================
# INITIAL FREE SURFACE
# ================================================================

def initial_free_surface(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    dam_radius: float,
) -> np.ndarray:

    radius = np.sqrt(
        X**2
        +
        Y**2
    )

    eta = np.where(
        radius
        <=
        dam_radius,
        2.0,
        1.0,
    )

    return eta


# ================================================================
# GET CELL CENTERS
# ================================================================

def get_state_centers(
    state,
):

    if hasattr(
        state,
        "p_centers",
    ):

        centers = (
            state.p_centers
        )

        if callable(
            centers
        ):

            centers = centers()

        return (
            np.asarray(
                centers[0]
            ),
            np.asarray(
                centers[1]
            ),
        )

    centers = (
        state.grid.p_centers
    )

    if callable(
        centers
    ):

        centers = centers()

    return (
        np.asarray(
            centers[0]
        ),
        np.asarray(
            centers[1]
        ),
    )


# ================================================================
# SOLUTION TIME
# ================================================================

def get_solution_time(
    solution,
) -> float:

    if hasattr(
        solution,
        "t",
    ):

        return float(
            solution.t
        )

    return float(
        solution.state.t
    )


# ================================================================
# EXTRACT q
# ================================================================

def extract_q(
    solution,
) -> np.ndarray:

    q = np.asarray(
        solution.state.q,
        dtype=np.float64,
    )

    # PyClaw:
    #     (3, nx, ny)
    #
    # CFO:
    #     (nx, ny, 3)

    return np.transpose(
        q,
        (
            1,
            2,
            0,
        ),
    )


# ================================================================
# CREATE SOLVER
# ================================================================

def create_solver():

    rp = (
        riemann
        .shallow_bathymetry_fwave_2D
    )

    solver = pyclaw.ClawSolver2D(
        rp
    )

    # ------------------------------------------------------------
    # This Riemann solver is used dimension-by-dimension.
    # ------------------------------------------------------------

    solver.dimensional_split = True

    solver.fwave = True

    # ------------------------------------------------------------
    # Limiter
    # ------------------------------------------------------------

    solver.limiters = (
        pyclaw.limiters.tvd.MC
    )

    # ------------------------------------------------------------
    # CFL
    # ------------------------------------------------------------

    solver.cfl_desired = 0.45

    solver.cfl_max = 0.50

    # ------------------------------------------------------------
    # Open / extrapolation boundaries
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

    return (
        solver,
        rp,
    )


# ================================================================
# RUN ONE CASE
# ================================================================

def simulate_case(
    *,
    case_name: str,
    amplitude: float,
    xc: float,
    yc: float,
    sigma: float,
    nx: int,
    ny: int,
    xlower: float,
    xupper: float,
    ylower: float,
    yupper: float,
    tfinal: float,
    num_snapshots: int,
    gravity: float,
    dam_radius: float,
):

    print(
        "\n========================================"
    )

    print(
        f"Running case: {case_name}"
    )

    print(
        "========================================"
    )

    # ============================================================
    # DOMAIN
    # ============================================================

    x_dimension = pyclaw.Dimension(
        xlower,
        xupper,
        nx,
        name="x",
    )

    y_dimension = pyclaw.Dimension(
        ylower,
        yupper,
        ny,
        name="y",
    )

    domain = pyclaw.Domain(
        [
            x_dimension,
            y_dimension,
        ]
    )

    # ============================================================
    # STATE
    #
    # 3 conserved variables
    # 1 auxiliary variable = bathymetry
    # ============================================================

    state = pyclaw.State(
        domain,
        3,
        1,
    )

    X, Y = get_state_centers(
        state
    )

    # ============================================================
    # BATHYMETRY
    # ============================================================

    if amplitude == 0.0:

        bathymetry = np.zeros_like(
            X,
            dtype=np.float64,
        )

    else:

        bathymetry = (
            gaussian_bathymetry(
                X,
                Y,
                amplitude=amplitude,
                xc=xc,
                yc=yc,
                sigma=sigma,
            )
        )

    # ============================================================
    # SAME INITIAL FREE SURFACE IN EVERY CASE
    # ============================================================

    eta_initial = (
        initial_free_surface(
            X,
            Y,
            dam_radius=dam_radius,
        )
    )

    # ------------------------------------------------------------
    # h = eta - b
    # ------------------------------------------------------------

    h_initial = (
        eta_initial
        -
        bathymetry
    )

    min_depth = float(
        np.min(
            h_initial
        )
    )

    if min_depth <= 1e-3:

        raise ValueError(
            f"Case {case_name}: "
            f"minimum initial depth "
            f"is too small: {min_depth}"
        )

    # ============================================================
    # INITIAL q
    # ============================================================

    state.q[
        0,
        :,
        :,
    ] = h_initial

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

    # ============================================================
    # AUXILIARY FIELD
    # ============================================================

    state.aux[
        0,
        :,
        :,
    ] = bathymetry

    # ============================================================
    # PHYSICAL PARAMETERS
    # ============================================================

    state.problem_data[
        "grav"
    ] = float(
        gravity
    )

    state.problem_data[
        "dry_tolerance"
    ] = 1e-3

    state.problem_data[
        "sea_level"
    ] = 0.0

    # ============================================================
    # SOLVER
    # ============================================================

    (
        solver,
        rp,
    ) = create_solver()

    # ------------------------------------------------------------
    # Explicitly populate the Fortran common parameters when this
    # Riemann implementation exposes set_cparam.
    # ------------------------------------------------------------

    if hasattr(
        rp,
        "set_cparam",
    ):

        rp.set_cparam(
            state.problem_data
        )

    # ============================================================
    # INITIAL SOLUTION
    # ============================================================

    solution = pyclaw.Solution(
        state,
        domain,
    )

    q_initial_copy = np.transpose(
        np.asarray(
            state.q,
            dtype=np.float64,
        ).copy(),
        (
            1,
            2,
            0,
        ),
    )

    # ============================================================
    # CONTROLLER
    # ============================================================

    controller = pyclaw.Controller()

    controller.solver = solver

    controller.solution = solution

    controller.tfinal = float(
        tfinal
    )

    # 50 output intervals -> 51 snapshots including t=0
    controller.num_output_times = (
        num_snapshots
        -
        1
    )

    controller.keep_copy = True

    # Do not create Clawpack fort.* files.
    controller.output_format = None

    # ============================================================
    # RUN
    # ============================================================

    status = controller.run()

    print(
        "PyClaw status:",
        status,
    )

    frames = list(
        controller.frames
    )

    # ============================================================
    # EXTRACT FRAMES
    # ============================================================

    frame_times = np.asarray(
        [
            get_solution_time(
                frame
            )
            for frame
            in frames
        ],
        dtype=np.float64,
    )

    frame_q = [
        extract_q(
            frame
        )
        for frame
        in frames
    ]

    # ------------------------------------------------------------
    # Depending on PyClaw version, keep_copy may or may not include
    # the initial state in controller.frames.
    # ------------------------------------------------------------

    if (
        len(
            frame_q
        )
        ==
        num_snapshots
    ):

        q = np.stack(
            frame_q,
            axis=0,
        )

        time = frame_times

    elif (
        len(
            frame_q
        )
        ==
        num_snapshots
        -
        1
    ):

        q = np.concatenate(
            [
                q_initial_copy[
                    None,
                    ...,
                ],
                np.stack(
                    frame_q,
                    axis=0,
                ),
            ],
            axis=0,
        )

        time = np.concatenate(
            [
                np.asarray(
                    [
                        0.0
                    ],
                    dtype=np.float64,
                ),
                frame_times,
            ]
        )

    else:

        raise RuntimeError(
            "Unexpected number of PyClaw frames. "
            f"Expected {num_snapshots} or "
            f"{num_snapshots - 1}, "
            f"got {len(frame_q)}."
        )

    # ============================================================
    # CHECK TIME
    # ============================================================

    expected_time = np.linspace(
        0.0,
        tfinal,
        num_snapshots,
        dtype=np.float64,
    )

    if not np.allclose(
        time,
        expected_time,
        atol=1e-8,
        rtol=1e-8,
    ):

        print(
            "Warning: PyClaw frame times differ "
            "slightly from target output times."
        )

    # ============================================================
    # CHECK OUTPUT
    # ============================================================

    if not np.all(
        np.isfinite(
            q
        )
    ):

        raise RuntimeError(
            f"{case_name} contains NaN/Inf."
        )

    if np.min(
        q[
            ...,
            0
        ]
    ) <= 0:

        raise RuntimeError(
            f"{case_name} produced "
            "non-positive depth."
        )

    eta0_check = (
        q[
            0,
            ...,
            0,
        ]
        +
        bathymetry
    )

    eta0_error = float(
        np.max(
            np.abs(
                eta0_check
                -
                eta_initial
            )
        )
    )

    print(
        f"q shape:       {q.shape}"
    )

    print(
        f"min h:         "
        f"{np.min(q[..., 0]):.6f}"
    )

    print(
        f"max |eta0-ref|:"
        f" {eta0_error:.3e}"
    )

    print(
        f"bed min/max:   "
        f"{bathymetry.min():.6f} / "
        f"{bathymetry.max():.6f}"
    )

    return {
        "q":
            q.astype(
                np.float32
            ),

        "bathymetry":
            bathymetry[
                ...,
                None
            ].astype(
                np.float32
            ),

        "eta_initial":
            eta_initial.astype(
                np.float32
            ),

        "time":
            expected_time.astype(
                np.float32
            ),

        "x":
            X[
                :,
                0
            ].astype(
                np.float32
            ),

        "y":
            Y[
                0,
                :
            ].astype(
                np.float32
            ),

        "amplitude":
            float(
                amplitude
            ),

        "xc":
            float(
                xc
            ),

        "yc":
            float(
                yc
            ),

        "sigma":
            float(
                sigma
            ),
    }


# ================================================================
# WRITE HDF5
# ================================================================

def write_dataset(
    output_path: Path,
    cases: dict,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    first_case = next(
        iter(
            cases.values()
        )
    )

    with h5py.File(
        output_path,
        "w",
    ) as h5:

        h5.create_dataset(
            "x",
            data=first_case[
                "x"
            ],
        )

        h5.create_dataset(
            "y",
            data=first_case[
                "y"
            ],
        )

        h5.create_dataset(
            "time",
            data=first_case[
                "time"
            ],
        )

        h5.attrs[
            "description"
        ] = (
            "Matched flat/Gaussian counterfactual "
            "test for bathymetry-conditioned CFO."
        )

        h5.attrs[
            "initial_condition"
        ] = (
            "Same eta in every case; "
            "h=eta-b; hu=hv=0."
        )

        cases_group = (
            h5.create_group(
                "cases"
            )
        )

        for (
            case_name,
            case,
        ) in cases.items():

            group = (
                cases_group.create_group(
                    case_name
                )
            )

            group.create_dataset(
                "q",
                data=case[
                    "q"
                ],
            )

            group.create_dataset(
                "bathymetry",
                data=case[
                    "bathymetry"
                ],
            )

            group.create_dataset(
                "eta_initial",
                data=case[
                    "eta_initial"
                ],
            )

            group.attrs[
                "amplitude"
            ] = case[
                "amplitude"
            ]

            group.attrs[
                "xc"
            ] = case[
                "xc"
            ]

            group.attrs[
                "yc"
            ] = case[
                "yc"
            ]

            group.attrs[
                "sigma"
            ] = case[
                "sigma"
            ]


# ================================================================
# MAIN
# ================================================================

def main() -> None:

    args = parse_args()

    output_path = Path(
        args.output
    )

    if not output_path.is_absolute():

        output_path = (
            PROJECT_ROOT
            /
            output_path
        )

    # ============================================================
    # CONTROLLED TERRAIN DEFINITIONS
    # ============================================================

    terrain_cases = {
        "flat": {
            "amplitude":
                0.0,

            "xc":
                0.0,

            "yc":
                0.0,
        },

        "hill_left": {
            "amplitude":
                args.hill_amplitude,

            "xc":
                -args.hill_offset,

            "yc":
                0.0,
        },

        "hill_center": {
            "amplitude":
                args.hill_amplitude,

            "xc":
                0.0,

            "yc":
                0.0,
        },

        "hill_right": {
            "amplitude":
                args.hill_amplitude,

            "xc":
                args.hill_offset,

            "yc":
                0.0,
        },
    }

    cases = {}

    for (
        case_name,
        parameters,
    ) in terrain_cases.items():

        cases[
            case_name
        ] = simulate_case(
            case_name=case_name,
            amplitude=parameters[
                "amplitude"
            ],
            xc=parameters[
                "xc"
            ],
            yc=parameters[
                "yc"
            ],
            sigma=args.hill_sigma,
            nx=args.nx,
            ny=args.ny,
            xlower=args.xlower,
            xupper=args.xupper,
            ylower=args.ylower,
            yupper=args.yupper,
            tfinal=args.tfinal,
            num_snapshots=(
                args.num_snapshots
            ),
            gravity=args.gravity,
            dam_radius=(
                args.dam_radius
            ),
        )

    # ============================================================
    # IMPORTANT MATCHED-ETA CHECK
    # ============================================================

    reference_eta = cases[
        "flat"
    ][
        "eta_initial"
    ]

    print(
        "\n========================================"
    )

    print(
        "MATCHED INITIAL FREE-SURFACE CHECK"
    )

    print(
        "========================================"
    )

    for (
        case_name,
        case,
    ) in cases.items():

        error = float(
            np.max(
                np.abs(
                    case[
                        "eta_initial"
                    ]
                    -
                    reference_eta
                )
            )
        )

        print(
            f"{case_name:<14}: "
            f"max |eta0-eta0_flat| "
            f"= {error:.3e}"
        )

    # ============================================================
    # SAVE
    # ============================================================

    write_dataset(
        output_path,
        cases,
    )

    print(
        "\n========================================"
    )

    print(
        "COUNTERFACTUAL DATASET COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        output_path
    )

    print(
        "\nCases:"
    )

    for case_name in cases:

        print(
            f"  {case_name}"
        )


if __name__ == "__main__":

    main()