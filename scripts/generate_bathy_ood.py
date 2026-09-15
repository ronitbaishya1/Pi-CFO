"""
Generate unseen-bathymetry shallow-water test cases.

IMPORTANT
---------
These cases are TEST ONLY.

They must NOT be used to retrain CFO, PI-CFO, or Bed-PI-CFO.

Purpose
-------
Test whether a model trained on single circular Gaussian hills learned:

    (q, b, t) -> q_t

as a more general bathymetry-conditioned operator,

or whether it only learned the original single-Gaussian terrain family.

Terrain families
----------------
1. id_gaussian
       Circular Gaussian inside the training distribution.
       Used only as a reference/control.

2. two_hills
       Sum of two separated Gaussian hills.

3. narrow_tall
       Circular Gaussian but narrower and taller than training.

4. elongated_ridge
       Axis-aligned anisotropic Gaussian.

5. rotated_ridge
       Anisotropic Gaussian rotated relative to x/y axes.

6. multi_hill
       Sum of three hills with different amplitudes and widths.

All cases use the SAME initial free-surface dam-break:

    eta = 2 inside radius
    eta = 1 outside radius

and

    h = eta - b
    hu = hv = 0

Output
------
data/shallow_water_bathy/swe_bathy_ood_32.h5

Run with
--------
conda activate swegen

python scripts/generate_bathy_ood.py
"""

from __future__ import annotations

import argparse
import json
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

def parse_args():

    parser = argparse.ArgumentParser()

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
        "--cases-per-family",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_ood_32.h5"
        ),
    )

    return parser.parse_args()


# ================================================================
# ROTATED / ANISOTROPIC GAUSSIAN
# ================================================================

def gaussian_component(
    X,
    Y,
    *,
    amplitude,
    xc,
    yc,
    sigma_x,
    sigma_y,
    angle_deg=0.0,
):

    theta = np.deg2rad(
        angle_deg
    )

    cos_theta = np.cos(
        theta
    )

    sin_theta = np.sin(
        theta
    )

    dx = (
        X
        -
        xc
    )

    dy = (
        Y
        -
        yc
    )

    # ------------------------------------------------------------
    # Coordinates in terrain's local rotated frame.
    # ------------------------------------------------------------

    x_rot = (
        cos_theta
        *
        dx
        +
        sin_theta
        *
        dy
    )

    y_rot = (
        -sin_theta
        *
        dx
        +
        cos_theta
        *
        dy
    )

    exponent = -0.5 * (
        (
            x_rot
            /
            sigma_x
        )
        ** 2
        +
        (
            y_rot
            /
            sigma_y
        )
        ** 2
    )

    return (
        amplitude
        *
        np.exp(
            exponent
        )
    )


# ================================================================
# TERRAIN FIELD
# ================================================================

def build_terrain(
    X,
    Y,
    terrain_spec,
):

    family = terrain_spec[
        "family"
    ]

    if family == "flat":

        return np.zeros_like(
            X,
            dtype=np.float64,
        )

    terrain = np.zeros_like(
        X,
        dtype=np.float64,
    )

    for component in terrain_spec[
        "components"
    ]:

        terrain += gaussian_component(
            X,
            Y,
            amplitude=component[
                "amplitude"
            ],
            xc=component[
                "xc"
            ],
            yc=component[
                "yc"
            ],
            sigma_x=component[
                "sigma_x"
            ],
            sigma_y=component[
                "sigma_y"
            ],
            angle_deg=component.get(
                "angle_deg",
                0.0,
            ),
        )

    return terrain


# ================================================================
# INITIAL FREE SURFACE
# ================================================================

def initial_eta(
    X,
    Y,
    dam_radius,
):

    radius = np.sqrt(
        X**2
        +
        Y**2
    )

    return np.where(
        radius
        <=
        dam_radius,
        2.0,
        1.0,
    )


# ================================================================
# CELL CENTERS
# ================================================================

def get_centers(
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
                centers[
                    0
                ]
            ),
            np.asarray(
                centers[
                    1
                ]
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
            centers[
                0
            ]
        ),
        np.asarray(
            centers[
                1
            ]
        ),
    )


# ================================================================
# SOLUTION TIME
# ================================================================

def solution_time(
    solution,
):

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
):

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
# SOLVER
# ================================================================

def create_solver():

    rp = (
        riemann
        .shallow_bathymetry_fwave_2D
    )

    solver = pyclaw.ClawSolver2D(
        rp
    )

    solver.dimensional_split = True

    solver.fwave = True

    solver.limiters = (
        pyclaw.limiters.tvd.MC
    )

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
# ONE SIMULATION
# ================================================================

def simulate_case(
    *,
    name,
    terrain_spec,
    args,
):

    print(
        "\n============================================"
    )

    print(
        f"CASE:   {name}"
    )

    print(
        f"FAMILY: {terrain_spec['family']}"
    )

    print(
        "============================================"
    )

    # ============================================================
    # DOMAIN
    # ============================================================

    x_dimension = pyclaw.Dimension(
        args.xlower,
        args.xupper,
        args.nx,
        name="x",
    )

    y_dimension = pyclaw.Dimension(
        args.ylower,
        args.yupper,
        args.ny,
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
    # ============================================================

    state = pyclaw.State(
        domain,
        3,
        1,
    )

    X, Y = get_centers(
        state
    )

    # ============================================================
    # TERRAIN
    # ============================================================

    bathymetry = build_terrain(
        X,
        Y,
        terrain_spec,
    )

    # ============================================================
    # SAME INITIAL FREE SURFACE
    # ============================================================

    eta0 = initial_eta(
        X,
        Y,
        args.dam_radius,
    )

    h0 = (
        eta0
        -
        bathymetry
    )

    minimum_depth = float(
        np.min(
            h0
        )
    )

    if minimum_depth <= 0.05:

        raise RuntimeError(
            f"{name}: depth too small. "
            f"min(h0) = {minimum_depth}"
        )

    # ============================================================
    # INITIAL STATE
    # ============================================================

    state.q[
        0
    ] = h0

    state.q[
        1
    ] = 0.0

    state.q[
        2
    ] = 0.0

    state.aux[
        0
    ] = bathymetry

    state.problem_data[
        "grav"
    ] = float(
        args.gravity
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

    solver, rp = create_solver()

    if hasattr(
        rp,
        "set_cparam",
    ):

        rp.set_cparam(
            state.problem_data
        )

    solution = pyclaw.Solution(
        state,
        domain,
    )

    q_initial = np.transpose(
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

    controller.solver = (
        solver
    )

    controller.solution = (
        solution
    )

    controller.tfinal = float(
        args.tfinal
    )

    controller.num_output_times = (
        args.num_snapshots
        -
        1
    )

    controller.keep_copy = True

    controller.output_format = None

    status = controller.run()

    print(
        "PyClaw status:",
        status,
    )

    frames = list(
        controller.frames
    )

    frame_q = [
        extract_q(
            frame
        )
        for frame
        in frames
    ]

    frame_time = np.asarray(
        [
            solution_time(
                frame
            )
            for frame
            in frames
        ],
        dtype=np.float64,
    )

    # ============================================================
    # HANDLE PYCLAW VERSION DIFFERENCES
    # ============================================================

    if (
        len(
            frame_q
        )
        ==
        args.num_snapshots
    ):

        q = np.stack(
            frame_q,
            axis=0,
        )

        time = (
            frame_time
        )

    elif (
        len(
            frame_q
        )
        ==
        args.num_snapshots
        -
        1
    ):

        q = np.concatenate(
            [
                q_initial[
                    None,
                    ...
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
                    ]
                ),
                frame_time,
            ]
        )

    else:

        raise RuntimeError(
            f"Unexpected number of frames: "
            f"{len(frame_q)}"
        )

    expected_time = np.linspace(
        0.0,
        args.tfinal,
        args.num_snapshots,
    )

    if not np.allclose(
        time,
        expected_time,
        atol=1e-8,
        rtol=1e-8,
    ):

        print(
            "Warning: frame times differ slightly "
            "from requested times."
        )

    # ============================================================
    # VALIDATION
    # ============================================================

    if not np.all(
        np.isfinite(
            q
        )
    ):

        raise RuntimeError(
            f"{name}: q contains NaN or Inf."
        )

    if np.min(
        q[
            ...,
            0
        ]
    ) <= 0:

        raise RuntimeError(
            f"{name}: non-positive depth."
        )

    eta_check = (
        q[
            0,
            ...,
            0
        ]
        +
        bathymetry
    )

    eta_error = float(
        np.max(
            np.abs(
                eta_check
                -
                eta0
            )
        )
    )

    print(
        f"b min/max: "
        f"{bathymetry.min():.5f} / "
        f"{bathymetry.max():.5f}"
    )

    print(
        f"initial h min: "
        f"{h0.min():.5f}"
    )

    print(
        f"all-time h min: "
        f"{q[..., 0].min():.5f}"
    )

    print(
        f"max initial eta error: "
        f"{eta_error:.3e}"
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
            eta0.astype(
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
    }


# ================================================================
# TERRAIN SPECIFICATIONS
# ================================================================

def terrain_specs():

    specs = {}

    # ============================================================
    # FLAT CONTROL
    # ============================================================

    specs[
        "flat"
    ] = {
        "family":
            "flat",

        "components":
            [],
    }

    # ============================================================
    # ID GAUSSIAN CONTROLS
    #
    # Inside original training ranges.
    # ============================================================

    id_parameters = [
        (
            0.15,
            -0.55,
            0.35,
            0.48,
        ),
        (
            0.20,
            0.45,
            -0.35,
            0.55,
        ),
        (
            0.12,
            0.10,
            0.65,
            0.42,
        ),
        (
            0.22,
            -0.30,
            -0.60,
            0.62,
        ),
    ]

    for index, (
        amplitude,
        xc,
        yc,
        sigma,
    ) in enumerate(
        id_parameters
    ):

        specs[
            f"id_gaussian_{index:02d}"
        ] = {
            "family":
                "id_gaussian",

            "components":
                [
                    {
                        "amplitude":
                            amplitude,

                        "xc":
                            xc,

                        "yc":
                            yc,

                        "sigma_x":
                            sigma,

                        "sigma_y":
                            sigma,

                        "angle_deg":
                            0.0,
                    }
                ],
        }

    # ============================================================
    # TWO HILLS
    # ============================================================

    two_hill_parameters = [
        [
            (
                0.15,
                -0.80,
                -0.20,
                0.45,
            ),
            (
                0.13,
                0.80,
                0.30,
                0.50,
            ),
        ],
        [
            (
                0.16,
                -0.65,
                0.55,
                0.40,
            ),
            (
                0.14,
                0.70,
                -0.55,
                0.48,
            ),
        ],
        [
            (
                0.12,
                -0.90,
                0.00,
                0.38,
            ),
            (
                0.18,
                0.60,
                0.10,
                0.55,
            ),
        ],
        [
            (
                0.15,
                -0.45,
                -0.65,
                0.50,
            ),
            (
                0.15,
                0.55,
                0.65,
                0.42,
            ),
        ],
    ]

    for index, hills in enumerate(
        two_hill_parameters
    ):

        components = []

        for (
            amplitude,
            xc,
            yc,
            sigma,
        ) in hills:

            components.append(
                {
                    "amplitude":
                        amplitude,

                    "xc":
                        xc,

                    "yc":
                        yc,

                    "sigma_x":
                        sigma,

                    "sigma_y":
                        sigma,

                    "angle_deg":
                        0.0,
                }
            )

        specs[
            f"two_hills_{index:02d}"
        ] = {
            "family":
                "two_hills",

            "components":
                components,
        }

    # ============================================================
    # NARROW / TALL GAUSSIAN
    #
    # Training:
    #     amplitude approx <= 0.25
    #     sigma approx >= 0.35
    #
    # These deliberately exceed those ranges.
    # ============================================================

    narrow_parameters = [
        (
            0.34,
            -0.60,
            0.20,
            0.24,
        ),
        (
            0.32,
            0.55,
            -0.40,
            0.22,
        ),
        (
            0.36,
            0.10,
            0.55,
            0.26,
        ),
        (
            0.30,
            -0.25,
            -0.60,
            0.20,
        ),
    ]

    for index, (
        amplitude,
        xc,
        yc,
        sigma,
    ) in enumerate(
        narrow_parameters
    ):

        specs[
            f"narrow_tall_{index:02d}"
        ] = {
            "family":
                "narrow_tall",

            "components":
                [
                    {
                        "amplitude":
                            amplitude,

                        "xc":
                            xc,

                        "yc":
                            yc,

                        "sigma_x":
                            sigma,

                        "sigma_y":
                            sigma,

                        "angle_deg":
                            0.0,
                    }
                ],
        }

    # ============================================================
    # ELONGATED RIDGE
    #
    # Never seen during circular-Gaussian training.
    # ============================================================

    elongated_parameters = [
        (
            0.22,
            -0.40,
            0.00,
            0.22,
            0.95,
        ),
        (
            0.20,
            0.45,
            -0.10,
            0.95,
            0.24,
        ),
        (
            0.24,
            0.00,
            0.45,
            0.25,
            0.85,
        ),
        (
            0.18,
            0.00,
            -0.50,
            0.90,
            0.28,
        ),
    ]

    for index, (
        amplitude,
        xc,
        yc,
        sigma_x,
        sigma_y,
    ) in enumerate(
        elongated_parameters
    ):

        specs[
            f"elongated_ridge_{index:02d}"
        ] = {
            "family":
                "elongated_ridge",

            "components":
                [
                    {
                        "amplitude":
                            amplitude,

                        "xc":
                            xc,

                        "yc":
                            yc,

                        "sigma_x":
                            sigma_x,

                        "sigma_y":
                            sigma_y,

                        "angle_deg":
                            0.0,
                    }
                ],
        }

    # ============================================================
    # ROTATED RIDGE
    # ============================================================

    rotated_parameters = [
        (
            0.22,
            0.00,
            0.00,
            0.22,
            0.90,
            30.0,
        ),
        (
            0.20,
            0.35,
            -0.30,
            0.25,
            0.95,
            45.0,
        ),
        (
            0.23,
            -0.40,
            0.25,
            0.24,
            0.85,
            60.0,
        ),
        (
            0.19,
            0.20,
            0.45,
            0.28,
            1.00,
            120.0,
        ),
    ]

    for index, (
        amplitude,
        xc,
        yc,
        sigma_x,
        sigma_y,
        angle,
    ) in enumerate(
        rotated_parameters
    ):

        specs[
            f"rotated_ridge_{index:02d}"
        ] = {
            "family":
                "rotated_ridge",

            "components":
                [
                    {
                        "amplitude":
                            amplitude,

                        "xc":
                            xc,

                        "yc":
                            yc,

                        "sigma_x":
                            sigma_x,

                        "sigma_y":
                            sigma_y,

                        "angle_deg":
                            angle,
                    }
                ],
        }

    # ============================================================
    # MULTI-HILL
    # ============================================================

    multi_parameters = [
        [
            (
                0.12,
                -0.85,
                -0.40,
                0.35,
            ),
            (
                0.15,
                0.00,
                0.65,
                0.45,
            ),
            (
                0.10,
                0.80,
                -0.25,
                0.32,
            ),
        ],
        [
            (
                0.13,
                -0.75,
                0.55,
                0.40,
            ),
            (
                0.11,
                0.00,
                -0.65,
                0.35,
            ),
            (
                0.16,
                0.75,
                0.45,
                0.50,
            ),
        ],
        [
            (
                0.10,
                -0.95,
                0.00,
                0.30,
            ),
            (
                0.17,
                0.00,
                0.00,
                0.48,
            ),
            (
                0.12,
                0.90,
                0.10,
                0.35,
            ),
        ],
        [
            (
                0.14,
                -0.55,
                -0.70,
                0.42,
            ),
            (
                0.12,
                -0.10,
                0.75,
                0.30,
            ),
            (
                0.13,
                0.75,
                -0.10,
                0.40,
            ),
        ],
    ]

    for index, hills in enumerate(
        multi_parameters
    ):

        components = []

        for (
            amplitude,
            xc,
            yc,
            sigma,
        ) in hills:

            components.append(
                {
                    "amplitude":
                        amplitude,

                    "xc":
                        xc,

                    "yc":
                        yc,

                    "sigma_x":
                        sigma,

                    "sigma_y":
                        sigma,

                    "angle_deg":
                        0.0,
                }
            )

        specs[
            f"multi_hill_{index:02d}"
        ] = {
            "family":
                "multi_hill",

            "components":
                components,
        }

    return specs


# ================================================================
# SAVE
# ================================================================

def save_dataset(
    output_path,
    results,
    specs,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    first = next(
        iter(
            results.values()
        )
    )

    with h5py.File(
        output_path,
        "w",
    ) as h5:

        h5.attrs[
            "description"
        ] = (
            "Frozen-model OOD bathymetry "
            "generalization benchmark."
        )

        h5.attrs[
            "warning"
        ] = (
            "TEST ONLY. Do not use for training."
        )

        h5.create_dataset(
            "x",
            data=first[
                "x"
            ],
        )

        h5.create_dataset(
            "y",
            data=first[
                "y"
            ],
        )

        h5.create_dataset(
            "time",
            data=first[
                "time"
            ],
        )

        cases_group = (
            h5.create_group(
                "cases"
            )
        )

        for (
            case_name,
            result,
        ) in results.items():

            group = (
                cases_group.create_group(
                    case_name
                )
            )

            group.create_dataset(
                "q",
                data=result[
                    "q"
                ],
            )

            group.create_dataset(
                "bathymetry",
                data=result[
                    "bathymetry"
                ],
            )

            group.create_dataset(
                "eta_initial",
                data=result[
                    "eta_initial"
                ],
            )

            group.attrs[
                "family"
            ] = specs[
                case_name
            ][
                "family"
            ]

            group.attrs[
                "terrain_spec_json"
            ] = json.dumps(
                specs[
                    case_name
                ]
            )


# ================================================================
# MAIN
# ================================================================

def main():

    args = parse_args()

    if (
        args.cases_per_family
        <
        1
        or
        args.cases_per_family
        >
        4
    ):

        raise ValueError(
            "--cases-per-family must "
            "be between 1 and 4."
        )

    output_path = Path(
        args.output
    )

    if not output_path.is_absolute():

        output_path = (
            PROJECT_ROOT
            /
            output_path
        )

    specs_all = terrain_specs()

    # ============================================================
    # SELECT FIRST N CASES FROM EACH FAMILY
    # ============================================================

    selected_specs = {
        "flat":
            specs_all[
                "flat"
            ]
    }

    families = [
        "id_gaussian",
        "two_hills",
        "narrow_tall",
        "elongated_ridge",
        "rotated_ridge",
        "multi_hill",
    ]

    for family in families:

        family_names = [
            name
            for name, spec
            in specs_all.items()
            if spec[
                "family"
            ]
            ==
            family
        ]

        family_names = sorted(
            family_names
        )[
            :
            args.cases_per_family
        ]

        for name in family_names:

            selected_specs[
                name
            ] = specs_all[
                name
            ]

    # ============================================================
    # GENERATE
    # ============================================================

    print(
        "\n============================================"
    )

    print(
        "OOD BATHYMETRY DATA GENERATION"
    )

    print(
        "============================================"
    )

    print(
        f"Cases per family: "
        f"{args.cases_per_family}"
    )

    print(
        f"Total trajectories: "
        f"{len(selected_specs)}"
    )

    print(
        "\nIMPORTANT: "
        "This dataset is TEST ONLY."
    )

    results = {}

    for (
        name,
        terrain_spec,
    ) in selected_specs.items():

        results[
            name
        ] = simulate_case(
            name=name,
            terrain_spec=terrain_spec,
            args=args,
        )

    save_dataset(
        output_path,
        results,
        selected_specs,
    )

    print(
        "\n============================================"
    )

    print(
        "OOD DATASET COMPLETE"
    )

    print(
        "============================================"
    )

    print(
        output_path
    )

    print(
        "\nFamilies:"
    )

    for family in [
        "flat"
    ] + families:

        count = sum(
            1
            for spec
            in selected_specs.values()
            if spec[
                "family"
            ]
            ==
            family
        )

        print(
            f"  {family:<18} "
            f"{count}"
        )


if __name__ == "__main__":

    main()