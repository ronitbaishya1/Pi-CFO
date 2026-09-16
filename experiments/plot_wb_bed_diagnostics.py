"""
Compare Bed-PI-CFO and WB-Bed-PI-CFO using the existing PyClaw
counterfactual and OOD bathymetry datasets.

Outputs
-------
1. 01_direct_condition_test.png
2. 02_isolated_hill_effect_error.png
3. 03_terrain_induced_speed_gaussian.png
4. 04_terrain_induced_speed_unseen.png

The comparison is:

    PyClaw / SWE expected
            vs
    Bed-PI-CFO
            vs
    WB-Bed-PI-CFO

Existing datasets
-----------------
data/shallow_water_bathy/gaussian_counterfactual_32.h5
data/shallow_water_bathy/swe_bathy_ood_32.h5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ================================================================
# PROJECT IMPORTS
# ================================================================

from models.fno import FNO2d

from bathy_bed_pi_cfo import (
    BathymetryBedRegularizedPICFO,
)

from wb_bathy_bed_pi_cfo import (
    WellBalancedBathymetryBedPICFO,
)

from train import (
    init_cfo_train_state,
)

from utils.checkpoints import (
    load_train_state,
)


# ================================================================
# CONSTANTS
# ================================================================

NX = 32
NY = 32

X_MIN = -2.5
X_MAX = 2.5

Y_MIN = -2.5
Y_MAX = 2.5

DX = 5.0 / 32.0
DY = 5.0 / 32.0

GRAVITY = 1.0

GAUSSIAN_CASES = [
    "hill_left",
    "hill_center",
    "hill_right",
]

OOD_CASES = [
    "two_hills",
    "narrow_tall",
    "elongated_ridge",
    "rotated_ridge",
    "multi_hill",
]


# ================================================================
# ARGUMENTS
# ================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--counterfactual-data",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "gaussian_counterfactual_32.h5"
        ),
    )

    parser.add_argument(
        "--ood-data",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_ood_32.h5"
        ),
    )

    parser.add_argument(
        "--bed-ckpt-dir",
        type=str,
        default=(
            "checkpoints/"
            "bathy_bed_lam07/"
            "seed0/"
            "best/"
            "bathy_bed_pi"
        ),
    )

    parser.add_argument(
        "--wb-ckpt-dir",
        type=str,
        default=(
            "checkpoints/"
            "wb_bathy_bed_pi/"
            "fno/"
            "lamwb_0p1/"
            "seed0/"
            "best"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/"
            "wb_bed_diagnostics"
        ),
    )

    parser.add_argument(
        "--lambda-pde",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--lambda-bed",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--lambda-wb",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--wb-eta0",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--plot-time",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--direct-case",
        type=str,
        default="hill_right",
    )

    return parser.parse_args()


# ================================================================
# PATH
# ================================================================

def resolve_path(path_string):

    path = Path(path_string)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


# ================================================================
# NORMALIZATION OF NAMES
# ================================================================

def normalize_name(value):

    value = str(value).lower()

    replacements = [
        ("-", "_"),
        (" ", "_"),
        ("/", "_"),
        ("\\", "_"),
        (".", "_"),
    ]

    for old, new in replacements:
        value = value.replace(old, new)

    while "__" in value:
        value = value.replace("__", "_")

    return value.strip("_")


# ================================================================
# HDF5 ARCHIVE
# ================================================================

class H5Archive:
    """
    Flexible reader for the two previously-generated HDF5 files.

    It supports the common layouts used in the project:

    1. groups named after terrain:
           hill_left/q
           hill_left/bathymetry
           ...

    2. explicit datasets:
           hill_left_q
           hill_left_bathymetry
           ...

    3. first dimension = terrain/case index together with a
       case_names / terrain_names string array.
    """

    def __init__(self, path):

        self.path = resolve_path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"HDF5 file does not exist:\n{self.path}"
            )

        self.file = h5py.File(
            self.path,
            "r",
        )

        self.datasets = {}

        def visitor(name, obj):

            if isinstance(
                obj,
                h5py.Dataset,
            ):
                self.datasets[name] = obj

        self.file.visititems(
            visitor
        )

        print()
        print("=" * 70)
        print("HDF5 DATASET")
        print(self.path)
        print("=" * 70)

        for name, dataset in self.datasets.items():

            print(
                f"{name:55s} "
                f"shape={dataset.shape}"
            )

        print("=" * 70)

    def close(self):

        try:
            self.file.close()
        except Exception:
            pass

    # ------------------------------------------------------------
    # ARRAY HELPERS
    # ------------------------------------------------------------

    def _array(self, path):

        return np.asarray(
            self.datasets[path]
        )

    def _state_paths(self):

        paths = []

        for path, ds in self.datasets.items():

            shape = ds.shape

            if len(shape) >= 4 and shape[-1] == 3:

                paths.append(
                    path
                )

        return paths

    def _bed_paths(self):

        paths = []

        for path, ds in self.datasets.items():

            shape = ds.shape

            if len(shape) < 2:
                continue

            if (
                len(shape) >= 4
                and
                shape[-1] == 3
            ):
                continue

            has_32_pair = False

            for i in range(
                len(shape) - 1
            ):

                if (
                    shape[i] == NX
                    and
                    shape[i + 1] == NY
                ):

                    has_32_pair = True
                    break

            if not has_32_pair:
                continue

            n = normalize_name(path)

            if any(
                token in n
                for token in [
                    "bathymetry",
                    "terrain",
                    "bed",
                    "_b",
                ]
            ):
                paths.append(
                    path
                )

        return paths

    def _time_paths(self):

        paths = []

        for path, ds in self.datasets.items():

            if len(ds.shape) != 1:
                continue

            if ds.shape[0] < 2:
                continue

            if ds.dtype.kind in {
                "S",
                "U",
                "O",
            }:
                continue

            n = normalize_name(path)

            if any(
                token in n
                for token in [
                    "time",
                    "times",
                    "_t",
                ]
            ):
                paths.append(
                    path
                )

        return paths

    # ------------------------------------------------------------
    # FIND CASE INDEX
    # ------------------------------------------------------------

    def _case_index(self, case_name):

        target = normalize_name(
            case_name
        )

        for path, ds in self.datasets.items():

            if len(ds.shape) != 1:
                continue

            if ds.dtype.kind not in {
                "S",
                "U",
                "O",
            }:
                continue

            try:

                values = np.asarray(
                    ds
                )

                decoded = []

                for value in values:

                    if isinstance(
                        value,
                        bytes,
                    ):

                        value = value.decode(
                            "utf-8"
                        )

                    decoded.append(
                        normalize_name(
                            value
                        )
                    )

                for index, value in enumerate(
                    decoded
                ):

                    if (
                        value == target
                        or
                        target in value
                        or
                        value in target
                    ):

                        return (
                            index,
                            len(decoded),
                            path,
                        )

            except Exception:

                continue

        return None

    # ------------------------------------------------------------
    # STATE SELECTION
    # ------------------------------------------------------------

    def _choose_state_path(
        self,
        paths,
        *,
        want_flat,
    ):

        state_paths = set(
            self._state_paths()
        )

        candidates = [
            p
            for p in paths
            if p in state_paths
        ]

        if len(
            candidates
        ) == 0:

            return None

        flat_tokens = [
            "flat",
            "no_bathy",
            "no_bed",
            "baseline",
        ]

        scored = []

        for path in candidates:

            n = normalize_name(
                path
            )

            is_flat = any(
                token in n
                for token in flat_tokens
            )

            if want_flat and not is_flat:
                continue

            if (
                not want_flat
                and
                is_flat
            ):
                continue

            score = 0

            if want_flat:
                score += 100

            for token, points in [
                ("q_flat", 30),
                ("flat_q", 30),
                ("q_terrain", 25),
                ("terrain_q", 25),
                ("q_hill", 25),
                ("hill_q", 25),
                ("solution", 10),
                ("trajectory", 10),
                ("state", 10),
            ]:

                if token in n:

                    score += points

            basename = normalize_name(
                Path(path).name
            )

            if basename == "q":
                score += 20

            scored.append(
                (
                    score,
                    path,
                )
            )

        if len(
            scored
        ) == 0:

            return None

        scored.sort(
            reverse=True
        )

        return scored[0][1]

    # ------------------------------------------------------------
    # BED SELECTION
    # ------------------------------------------------------------

    def _choose_bed_path(
        self,
        paths,
    ):

        bed_paths = set(
            self._bed_paths()
        )

        candidates = [
            p
            for p in paths
            if p in bed_paths
        ]

        if not candidates:
            return None

        scored = []

        for path in candidates:

            n = normalize_name(
                path
            )

            score = 0

            if "bathymetry" in n:
                score += 40

            if "terrain" in n:
                score += 30

            if "bed" in n:
                score += 25

            basename = normalize_name(
                Path(path).name
            )

            if basename == "b":
                score += 50

            scored.append(
                (
                    score,
                    path,
                )
            )

        scored.sort(
            reverse=True
        )

        return scored[0][1]

    # ------------------------------------------------------------
    # NORMALIZE STATE
    # ------------------------------------------------------------

    def _normalize_q(
        self,
        array,
        *,
        sample_index,
    ):

        q = np.asarray(
            array,
            dtype=np.float32,
        )

        while q.ndim > 4:

            index = min(
                sample_index,
                q.shape[0] - 1,
            )

            q = q[
                index
            ]

        if (
            q.ndim == 4
            and
            q.shape[-1] == 3
        ):

            return q

        raise ValueError(
            "Could not normalize state trajectory.\n"
            f"Shape = {q.shape}"
        )

    # ------------------------------------------------------------
    # NORMALIZE BED
    # ------------------------------------------------------------

    def _normalize_b(
        self,
        array,
        *,
        sample_index,
    ):

        b = np.asarray(
            array,
            dtype=np.float32,
        )

        while b.ndim > 3:

            index = min(
                sample_index,
                b.shape[0] - 1,
            )

            b = b[
                index
            ]

        if (
            b.ndim == 3
            and
            b.shape[-1] == 1
        ):

            b = b[
                ...,
                0
            ]

        elif (
            b.ndim == 3
            and
            b.shape[0] != NX
        ):

            index = min(
                sample_index,
                b.shape[0] - 1,
            )

            b = b[
                index
            ]

        if b.shape != (
            NX,
            NY,
        ):

            raise ValueError(
                "Could not normalize bathymetry.\n"
                f"Shape = {b.shape}"
            )

        return b

    # ------------------------------------------------------------
    # APPLY CASE INDEX
    # ------------------------------------------------------------

    def _slice_case_index(
        self,
        array,
        *,
        case_info,
    ):

        if case_info is None:

            return array

        index, num_cases, _ = case_info

        if (
            array.ndim >= 1
            and
            array.shape[0] == num_cases
        ):

            return array[
                index
            ]

        return array

    # ------------------------------------------------------------
    # GLOBAL FLAT STATE
    # ------------------------------------------------------------

    def _find_global_flat_state(
        self,
        *,
        case_info,
        sample_index,
    ):

        paths = self._state_paths()

        path = self._choose_state_path(
            paths,
            want_flat=True,
        )

        if path is None:
            return None

        array = self._array(
            path
        )

        array = self._slice_case_index(
            array,
            case_info=case_info,
        )

        return self._normalize_q(
            array,
            sample_index=sample_index,
        )

    # ------------------------------------------------------------
    # GET TIME
    # ------------------------------------------------------------

    def _get_time(
        self,
        *,
        case_paths,
        trajectory_length,
    ):

        time_paths = self._time_paths()

        # Prefer case-specific time
        local = [
            p
            for p in time_paths
            if p in case_paths
        ]

        candidates = (
            local
            if local
            else time_paths
        )

        for path in candidates:

            t = np.asarray(
                self.datasets[path],
                dtype=np.float32,
            )

            if len(t) == trajectory_length:

                return t

        return np.linspace(
            0.0,
            1.0,
            trajectory_length,
            dtype=np.float32,
        )

    # ------------------------------------------------------------
    # GET COUNTERFACTUAL PAIR
    # ------------------------------------------------------------

    def get_pair(
        self,
        case_name,
        *,
        sample_index=0,
    ):

        target = normalize_name(
            case_name
        )

        case_paths = [
            path
            for path in self.datasets
            if target in normalize_name(
                path
            )
        ]

        case_info = self._case_index(
            case_name
        )

        # --------------------------------------------------------
        # If cases are indexed along first dimension
        # --------------------------------------------------------

        indexed_paths = []

        if case_info is not None:

            _, num_cases, _ = case_info

            for path, ds in self.datasets.items():

                if (
                    len(ds.shape) >= 1
                    and
                    ds.shape[0] == num_cases
                ):

                    indexed_paths.append(
                        path
                    )

        candidate_paths = list(
            dict.fromkeys(
                case_paths
                +
                indexed_paths
            )
        )

        # --------------------------------------------------------
        # Terrain trajectory
        # --------------------------------------------------------

        terrain_q_path = self._choose_state_path(
            candidate_paths,
            want_flat=False,
        )

        if terrain_q_path is None:

            raise RuntimeError(
                "\nCould not find terrain state for "
                f"case '{case_name}'.\n"
                "Look at the printed HDF5 inventory above."
            )

        terrain_q = self._array(
            terrain_q_path
        )

        terrain_q = self._slice_case_index(
            terrain_q,
            case_info=case_info,
        )

        terrain_q = self._normalize_q(
            terrain_q,
            sample_index=sample_index,
        )

        # --------------------------------------------------------
        # Flat trajectory
        # --------------------------------------------------------

        flat_q_path = self._choose_state_path(
            candidate_paths,
            want_flat=True,
        )

        flat_q = None

        if flat_q_path is not None:

            flat_q = self._array(
                flat_q_path
            )

            flat_q = self._slice_case_index(
                flat_q,
                case_info=case_info,
            )

            flat_q = self._normalize_q(
                flat_q,
                sample_index=sample_index,
            )

        if flat_q is None:

            flat_q = self._find_global_flat_state(
                case_info=case_info,
                sample_index=sample_index,
            )

        if flat_q is None:

            raise RuntimeError(
                "\nCould not find flat reference trajectory "
                f"for '{case_name}'.\n"
                "The script requires the paired PyClaw flat "
                "trajectory used by the counterfactual benchmark."
            )

        # --------------------------------------------------------
        # Bathymetry
        # --------------------------------------------------------

        bed_path = self._choose_bed_path(
            candidate_paths
        )

        if bed_path is None:

            raise RuntimeError(
                "\nCould not find bathymetry for "
                f"'{case_name}'."
            )

        bed = self._array(
            bed_path
        )

        bed = self._slice_case_index(
            bed,
            case_info=case_info,
        )

        bed = self._normalize_b(
            bed,
            sample_index=sample_index,
        )

        # --------------------------------------------------------
        # Time
        # --------------------------------------------------------

        t = self._get_time(
            case_paths=candidate_paths,
            trajectory_length=terrain_q.shape[0],
        )

        if (
            terrain_q.shape
            !=
            flat_q.shape
        ):

            raise ValueError(
                f"Terrain/flat state shapes differ for "
                f"{case_name}:\n"
                f"terrain={terrain_q.shape}\n"
                f"flat={flat_q.shape}"
            )

        print()
        print(
            f"[{case_name}]"
        )

        print(
            "terrain q:",
            terrain_q_path,
            terrain_q.shape,
        )

        print(
            "flat q:",
            flat_q_path
            if flat_q_path is not None
            else "global flat",
            flat_q.shape,
        )

        print(
            "bathymetry:",
            bed_path,
            bed.shape,
        )

        print(
            "time:",
            t.shape,
            f"{t[0]:.3f} -> {t[-1]:.3f}",
        )

        return {
            "terrain_q": terrain_q,
            "flat_q": flat_q,
            "b": bed,
            "time": t,
        }


# ================================================================
# FNO
# ================================================================

def build_fno():

    return FNO2d(
        num_channels=3,
        modes1=12,
        modes2=12,
        width=64,
        num_blocks=4,
        use_condition=True,
        use_time=True,
    )


# ================================================================
# ORIGINAL BED-PI-CFO
# ================================================================

def build_bed_method(
    lambda_pde,
    lambda_bed,
):

    model = build_fno()

    return BathymetryBedRegularizedPICFO(
        model=model,
        input_shape=(
            NX,
            NY,
            3,
        ),
        condition_shape=(
            NX,
            NY,
            1,
        ),
        gamma=1e-5,
        spline_type="quintic",
        lambda_pde=lambda_pde,
        lambda_bed=lambda_bed,
        dx=DX,
        dy=DY,
        gravity=GRAVITY,
    )


# ================================================================
# WB BED-PI-CFO
# ================================================================

def build_wb_method(
    lambda_pde,
    lambda_bed,
    lambda_wb,
    wb_eta0,
):

    model = build_fno()

    return WellBalancedBathymetryBedPICFO(
        model=model,
        input_shape=(
            NX,
            NY,
            3,
        ),
        condition_shape=(
            NX,
            NY,
            1,
        ),
        gamma=1e-5,
        spline_type="quintic",
        lambda_pde=lambda_pde,
        lambda_bed=lambda_bed,
        lambda_wb=lambda_wb,
        wb_eta0=wb_eta0,
        dx=DX,
        dy=DY,
        gravity=GRAVITY,
    )


# ================================================================
# RESTORE
# ================================================================

def restore_checkpoint(
    method,
    checkpoint_manager_dir,
):

    path = resolve_path(
        checkpoint_manager_dir
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Checkpoint path does not exist:\n{path}"
        )

    root = path.parent

    prefix = path.name

    target_state = init_cfo_train_state(
        method,
        seed=0,
        learning_rate=1e-4,
        beta1=0.9,
        beta2=0.99,
    )

    state = load_train_state(
        target_state,
        ckpt_dir=str(
            root
        ),
        prefix=prefix,
        step=None,
        max_to_keep=1,
    )

    return state


# ================================================================
# MODEL ROLLOUT
# ================================================================

def rollout(
    method,
    state,
    q0,
    b,
    *,
    trajectory_points,
    steps_per_segment,
):

    q0 = np.asarray(
        q0,
        dtype=np.float32,
    )

    b = np.asarray(
        b,
        dtype=np.float32,
    )

    q0_batch = jnp.asarray(
        q0[
            None,
            ...
        ],
        dtype=jnp.float32,
    )

    condition = jnp.asarray(
        b[
            None,
            ...,
            None,
        ],
        dtype=jnp.float32,
    )

    prediction = method.uniform_inference(
        state,
        q0_batch,
        trajectory_points_num=trajectory_points,
        steps_per_segment=steps_per_segment,
        condition=condition,
        method="RK4",
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )

    if (
        prediction.ndim == 5
        and
        prediction.shape[0] == 1
    ):

        prediction = prediction[
            0
        ]

    elif (
        prediction.ndim == 5
        and
        prediction.shape[1] == 1
    ):

        prediction = prediction[
            :,
            0,
            ...,
        ]

    if prediction.ndim != 4:

        raise ValueError(
            "Unexpected rollout shape: "
            f"{prediction.shape}"
        )

    return prediction


# ================================================================
# ROLLOUT PAIR
# ================================================================

def model_pair_rollout(
    method,
    state,
    data,
    *,
    steps_per_segment,
):

    q_hill_true = data[
        "terrain_q"
    ]

    q_flat_true = data[
        "flat_q"
    ]

    b = data[
        "b"
    ]

    num_t = q_hill_true.shape[
        0
    ]

    q_hill_pred = rollout(
        method,
        state,
        q_hill_true[0],
        b,
        trajectory_points=num_t,
        steps_per_segment=steps_per_segment,
    )

    q_flat_pred = rollout(
        method,
        state,
        q_flat_true[0],
        np.zeros_like(
            b
        ),
        trajectory_points=num_t,
        steps_per_segment=steps_per_segment,
    )

    return (
        q_hill_pred,
        q_flat_pred,
    )


# ================================================================
# VELOCITY
# ================================================================

def speed_from_q(
    q,
):

    h = np.maximum(
        q[
            ...,
            0
        ],
        1e-6,
    )

    hu = q[
        ...,
        1
    ]

    hv = q[
        ...,
        2
    ]

    u = hu / h
    v = hv / h

    return np.sqrt(
        u**2
        +
        v**2
    )


# ================================================================
# TERRAIN-INDUCED SPEED
# ================================================================

def terrain_speed_effect(
    q_terrain,
    q_flat,
    time_index,
):

    terrain_speed = speed_from_q(
        q_terrain[
            time_index
        ]
    )

    flat_speed = speed_from_q(
        q_flat[
            time_index
        ]
    )

    return (
        terrain_speed
        -
        flat_speed
    )


# ================================================================
# RELATIVE TERRAIN-EFFECT ERROR
# ================================================================

def relative_effect_error(
    true_terrain,
    true_flat,
    pred_terrain,
    pred_flat,
):

    true_effect = (
        true_terrain
        -
        true_flat
    )

    pred_effect = (
        pred_terrain
        -
        pred_flat
    )

    num_t = true_effect.shape[
        0
    ]

    error = np.zeros(
        num_t,
        dtype=np.float64,
    )

    for i in range(
        num_t
    ):

        truth = true_effect[
            i
        ]

        prediction = pred_effect[
            i
        ]

        numerator = np.linalg.norm(
            prediction
            -
            truth
        )

        denominator = np.linalg.norm(
            truth
        )

        if denominator < 1e-10:

            if numerator < 1e-10:
                error[i] = 0.0
            else:
                error[i] = numerator

        else:

            error[i] = (
                numerator
                /
                denominator
            )

    return error


# ================================================================
# CENTRAL DIFFERENCE
# ================================================================

def central_diff_x(
    field,
):

    return (
        np.roll(
            field,
            -1,
            axis=0,
        )
        -
        np.roll(
            field,
            1,
            axis=0,
        )
    ) / (
        2.0
        *
        DX
    )


def central_diff_y(
    field,
):

    return (
        np.roll(
            field,
            -1,
            axis=1,
        )
        -
        np.roll(
            field,
            1,
            axis=1,
        )
    ) / (
        2.0
        *
        DY
    )


# ================================================================
# DIRECT MODEL VECTOR FIELD
# ================================================================

def model_vector_field(
    method,
    state,
    q,
    b,
    time_value,
):

    q_batch = jnp.asarray(
        q[
            None,
            ...
        ],
        dtype=jnp.float32,
    )

    b_batch = jnp.asarray(
        b[
            None,
            ...,
            None,
        ],
        dtype=jnp.float32,
    )

    t_batch = jnp.asarray(
        [
            float(
                time_value
            )
        ],
        dtype=jnp.float32,
    )

    vector_field = method._model_apply(
        state.params,
        q_batch,
        t_batch,
        b_batch,
    )

    vector_field = np.asarray(
        vector_field,
        dtype=np.float32,
    )

    return vector_field[
        0
    ]


# ================================================================
# COORDINATES
# ================================================================

def coordinates():

    x = np.linspace(
        X_MIN,
        X_MAX,
        NX,
    )

    y = np.linspace(
        Y_MIN,
        Y_MAX,
        NY,
    )

    return (
        x,
        y,
    )


# ================================================================
# ADD TERRAIN CONTOURS
# ================================================================

def add_terrain_contours(
    ax,
    b,
):

    x, y = coordinates()

    bmax = float(
        np.max(
            b
        )
    )

    if bmax <= 1e-12:
        return

    positive = b[
        b > 0
    ]

    if positive.size == 0:
        return

    levels = np.linspace(
        max(
            float(
                np.min(
                    positive
                )
            ),
            0.10
            *
            bmax,
        ),
        bmax,
        6,
    )

    ax.contour(
        x,
        y,
        b.T,
        levels=levels,
        linewidths=0.8,
    )


# ================================================================
# HEATMAP
# ================================================================

def heatmap(
    ax,
    field,
    *,
    vlim,
    title,
    b=None,
    cmap="coolwarm",
):

    x, y = coordinates()

    image = ax.imshow(
        field.T,
        origin="lower",
        extent=[
            X_MIN,
            X_MAX,
            Y_MIN,
            Y_MAX,
        ],
        cmap=cmap,
        vmin=-vlim,
        vmax=vlim,
        interpolation="nearest",
        aspect="equal",
    )

    if b is not None:

        add_terrain_contours(
            ax,
            b,
        )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "x"
    )

    ax.set_ylabel(
        "y"
    )

    return image


# ================================================================
# PLOT 1 — DIRECT CONDITION TEST
# ================================================================

def plot_direct_condition_test(
    *,
    data,
    bed_method,
    bed_state,
    wb_method,
    wb_state,
    plot_time,
    output_path,
):

    time = data[
        "time"
    ]

    index = int(
        np.argmin(
            np.abs(
                time
                -
                plot_time
            )
        )
    )

    t_value = float(
        time[
            index
        ]
    )

    q_same = data[
        "terrain_q"
    ][
        index
    ]

    b = data[
        "b"
    ]

    b_zero = np.zeros_like(
        b
    )

    # ------------------------------------------------------------
    # SWE expected response to changing only b
    #
    # q_t momentum contribution from bathymetry:
    #
    # (hu)_t = ... - g h b_x
    # (hv)_t = ... - g h b_y
    # ------------------------------------------------------------

    h = np.maximum(
        q_same[
            ...,
            0
        ],
        1e-6,
    )

    db_dx = central_diff_x(
        b
    )

    db_dy = central_diff_y(
        b
    )

    expected_hu = (
        -GRAVITY
        *
        h
        *
        db_dx
    )

    expected_hv = (
        -GRAVITY
        *
        h
        *
        db_dy
    )

    # ------------------------------------------------------------
    # Bed-PI-CFO
    # ------------------------------------------------------------

    bed_with = model_vector_field(
        bed_method,
        bed_state,
        q_same,
        b,
        t_value,
    )

    bed_flat = model_vector_field(
        bed_method,
        bed_state,
        q_same,
        b_zero,
        t_value,
    )

    bed_delta = (
        bed_with
        -
        bed_flat
    )

    # ------------------------------------------------------------
    # WB-Bed-PI-CFO
    # ------------------------------------------------------------

    wb_with = model_vector_field(
        wb_method,
        wb_state,
        q_same,
        b,
        t_value,
    )

    wb_flat = model_vector_field(
        wb_method,
        wb_state,
        q_same,
        b_zero,
        t_value,
    )

    wb_delta = (
        wb_with
        -
        wb_flat
    )

    hu_fields = [
        expected_hu,
        bed_delta[
            ...,
            1
        ],
        wb_delta[
            ...,
            1
        ],
    ]

    hv_fields = [
        expected_hv,
        bed_delta[
            ...,
            2
        ],
        wb_delta[
            ...,
            2
        ],
    ]

    hu_lim = max(
        max(
            float(
                np.max(
                    np.abs(
                        value
                    )
                )
            )
            for value in hu_fields
        ),
        1e-8,
    )

    hv_lim = max(
        max(
            float(
                np.max(
                    np.abs(
                        value
                    )
                )
            )
            for value in hv_fields
        ),
        1e-8,
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            16,
            10,
        ),
    )

    column_names = [
        "SWE expected",
        "Bed-PI-CFO",
        "WB-Bed-PI-CFO",
    ]

    for col in range(
        3
    ):

        im = heatmap(
            axes[
                0,
                col,
            ],
            hu_fields[
                col
            ],
            vlim=hu_lim,
            title=(
                column_names[
                    col
                ]
                +
                "\n"
                +
                r"$\Delta(hu)_t$"
            ),
            b=b,
        )

        fig.colorbar(
            im,
            ax=axes[
                0,
                col,
            ],
            shrink=0.85,
        )

        im = heatmap(
            axes[
                1,
                col,
            ],
            hv_fields[
                col
            ],
            vlim=hv_lim,
            title=(
                column_names[
                    col
                ]
                +
                "\n"
                +
                r"$\Delta(hv)_t$"
            ),
            b=b,
        )

        fig.colorbar(
            im,
            ax=axes[
                1,
                col,
            ],
            shrink=0.85,
        )

    fig.suptitle(
        (
            "Direct condition test: "
            "same q and t, change only bathymetry\n"
            f"t = {t_value:.2f}"
        ),
        fontsize=16,
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ================================================================
# PLOT 2 — ISOLATED HILL EFFECT ERROR
# ================================================================

def plot_isolated_hill_errors(
    *,
    gaussian_results,
    output_path,
    csv_path,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            18,
            5.5,
        ),
    )

    csv_rows = []

    for column, case_name in enumerate(
        GAUSSIAN_CASES
    ):

        result = gaussian_results[
            case_name
        ]

        data = result[
            "data"
        ]

        time = data[
            "time"
        ]

        bed_error = relative_effect_error(
            data[
                "terrain_q"
            ],
            data[
                "flat_q"
            ],
            result[
                "bed_hill"
            ],
            result[
                "bed_flat"
            ],
        )

        wb_error = relative_effect_error(
            data[
                "terrain_q"
            ],
            data[
                "flat_q"
            ],
            result[
                "wb_hill"
            ],
            result[
                "wb_flat"
            ],
        )

        ax = axes[
            column
        ]

        ax.plot(
            time,
            bed_error,
            linewidth=2,
            label="Bed-PI-CFO",
        )

        ax.plot(
            time,
            wb_error,
            linewidth=2,
            label="WB-Bed-PI-CFO",
        )

        ax.set_title(
            case_name
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            (
                "Relative error in "
                "terrain-induced Δq"
            )
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend()

        for i in range(
            len(
                time
            )
        ):

            csv_rows.append(
                {
                    "case": case_name,
                    "time": float(
                        time[
                            i
                        ]
                    ),
                    "bed_pi_cfo_error": float(
                        bed_error[
                            i
                        ]
                    ),
                    "wb_bed_pi_cfo_error": float(
                        wb_error[
                            i
                        ]
                    ),
                }
            )

    fig.suptitle(
        "Error in the isolated Gaussian-hill effect",
        fontsize=16,
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case",
                "time",
                "bed_pi_cfo_error",
                "wb_bed_pi_cfo_error",
            ],
        )

        writer.writeheader()
        writer.writerows(
            csv_rows
        )


# ================================================================
# PLOT 3 — GAUSSIAN TERRAIN-INDUCED SPEED
# ================================================================

def plot_gaussian_speed(
    *,
    gaussian_results,
    plot_time,
    output_path,
):

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(
            16,
            16,
        ),
    )

    model_names = [
        "PyClaw",
        "Bed-PI-CFO",
        "WB-Bed-PI-CFO",
    ]

    for row, case_name in enumerate(
        GAUSSIAN_CASES
    ):

        result = gaussian_results[
            case_name
        ]

        data = result[
            "data"
        ]

        time = data[
            "time"
        ]

        index = int(
            np.argmin(
                np.abs(
                    time
                    -
                    plot_time
                )
            )
        )

        fields = [
            terrain_speed_effect(
                data[
                    "terrain_q"
                ],
                data[
                    "flat_q"
                ],
                index,
            ),
            terrain_speed_effect(
                result[
                    "bed_hill"
                ],
                result[
                    "bed_flat"
                ],
                index,
            ),
            terrain_speed_effect(
                result[
                    "wb_hill"
                ],
                result[
                    "wb_flat"
                ],
                index,
            ),
        ]

        vlim = max(
            max(
                float(
                    np.max(
                        np.abs(
                            field
                        )
                    )
                )
                for field in fields
            ),
            1e-8,
        )

        for col in range(
            3
        ):

            im = heatmap(
                axes[
                    row,
                    col,
                ],
                fields[
                    col
                ],
                vlim=vlim,
                title=(
                    case_name
                    +
                    "\n"
                    +
                    model_names[
                        col
                    ]
                ),
                b=data[
                    "b"
                ],
            )

            fig.colorbar(
                im,
                ax=axes[
                    row,
                    col,
                ],
                shrink=0.80,
                label=(
                    r"$|\mathbf{u}|_{terrain}"
                    r"-|\mathbf{u}|_{flat}$"
                ),
            )

    fig.suptitle(
        (
            "Terrain-induced speed for Gaussian hills "
            f"at t = {plot_time:.2f}"
        ),
        fontsize=17,
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.965,
        ]
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ================================================================
# PLOT 4 — UNSEEN TERRAIN SPEED
# ================================================================

def plot_unseen_speed(
    *,
    ood_results,
    plot_time,
    output_path,
):

    fig, axes = plt.subplots(
        len(
            OOD_CASES
        ),
        3,
        figsize=(
            16,
            24,
        ),
    )

    model_names = [
        "PyClaw",
        "Bed-PI-CFO",
        "WB-Bed-PI-CFO",
    ]

    for row, case_name in enumerate(
        OOD_CASES
    ):

        result = ood_results[
            case_name
        ]

        data = result[
            "data"
        ]

        time = data[
            "time"
        ]

        index = int(
            np.argmin(
                np.abs(
                    time
                    -
                    plot_time
                )
            )
        )

        fields = [
            terrain_speed_effect(
                data[
                    "terrain_q"
                ],
                data[
                    "flat_q"
                ],
                index,
            ),
            terrain_speed_effect(
                result[
                    "bed_hill"
                ],
                result[
                    "bed_flat"
                ],
                index,
            ),
            terrain_speed_effect(
                result[
                    "wb_hill"
                ],
                result[
                    "wb_flat"
                ],
                index,
            ),
        ]

        # --------------------------------------------------------
        # Same color scale across PyClaw, Bed, WB FOR EACH TERRAIN
        # --------------------------------------------------------

        vlim = max(
            max(
                float(
                    np.max(
                        np.abs(
                            field
                        )
                    )
                )
                for field in fields
            ),
            1e-8,
        )

        for col in range(
            3
        ):

            im = heatmap(
                axes[
                    row,
                    col,
                ],
                fields[
                    col
                ],
                vlim=vlim,
                title=(
                    case_name
                    +
                    "\n"
                    +
                    model_names[
                        col
                    ]
                ),
                b=data[
                    "b"
                ],
            )

            fig.colorbar(
                im,
                ax=axes[
                    row,
                    col,
                ],
                shrink=0.78,
                label=(
                    r"$|\mathbf{u}|_{terrain}"
                    r"-|\mathbf{u}|_{flat}$"
                ),
            )

    fig.suptitle(
        (
            "Terrain-induced speed on unseen bathymetry "
            f"at t = {plot_time:.2f}"
        ),
        fontsize=18,
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.975,
        ]
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ================================================================
# EVALUATE ONE CASE
# ================================================================

def evaluate_case(
    *,
    archive,
    case_name,
    bed_method,
    bed_state,
    wb_method,
    wb_state,
    sample_index,
    steps_per_segment,
):

    data = archive.get_pair(
        case_name,
        sample_index=sample_index,
    )

    print()
    print(
        f"Running Bed-PI-CFO: {case_name}"
    )

    (
        bed_hill,
        bed_flat,
    ) = model_pair_rollout(
        bed_method,
        bed_state,
        data,
        steps_per_segment=steps_per_segment,
    )

    print(
        f"Running WB-Bed-PI-CFO: {case_name}"
    )

    (
        wb_hill,
        wb_flat,
    ) = model_pair_rollout(
        wb_method,
        wb_state,
        data,
        steps_per_segment=steps_per_segment,
    )

    return {
        "data": data,
        "bed_hill": bed_hill,
        "bed_flat": bed_flat,
        "wb_hill": wb_hill,
        "wb_flat": wb_flat,
    }


# ================================================================
# MAIN
# ================================================================

def main():

    args = parse_args()

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 72)
    print("WB-BED-PI-CFO COUNTERFACTUAL DIAGNOSTICS")
    print("=" * 72)

    print(
        "lambda_PDE:",
        args.lambda_pde,
    )

    print(
        "lambda_bed:",
        args.lambda_bed,
    )

    print(
        "lambda_WB:",
        args.lambda_wb,
    )

    print(
        "Plot time:",
        args.plot_time,
    )

    print("=" * 72)

    # ============================================================
    # METHODS
    # ============================================================

    bed_method = build_bed_method(
        args.lambda_pde,
        args.lambda_bed,
    )

    wb_method = build_wb_method(
        args.lambda_pde,
        args.lambda_bed,
        args.lambda_wb,
        args.wb_eta0,
    )

    # ============================================================
    # CHECKPOINTS
    # ============================================================

    print()
    print(
        "Restoring Bed-PI-CFO..."
    )

    bed_state = restore_checkpoint(
        bed_method,
        args.bed_ckpt_dir,
    )

    print(
        "Bed-PI-CFO restored."
    )

    print()
    print(
        "Restoring WB-Bed-PI-CFO..."
    )

    wb_state = restore_checkpoint(
        wb_method,
        args.wb_ckpt_dir,
    )

    print(
        "WB-Bed-PI-CFO restored."
    )

    # ============================================================
    # ARCHIVES
    # ============================================================

    gaussian_archive = H5Archive(
        args.counterfactual_data
    )

    ood_archive = H5Archive(
        args.ood_data
    )

    try:

        # ========================================================
        # GAUSSIAN CASES
        # ========================================================

        gaussian_results = {}

        for case_name in GAUSSIAN_CASES:

            gaussian_results[
                case_name
            ] = evaluate_case(
                archive=gaussian_archive,
                case_name=case_name,
                bed_method=bed_method,
                bed_state=bed_state,
                wb_method=wb_method,
                wb_state=wb_state,
                sample_index=args.sample_index,
                steps_per_segment=(
                    args.steps_per_segment
                ),
            )

        # ========================================================
        # OOD CASES
        # ========================================================

        ood_results = {}

        for case_name in OOD_CASES:

            ood_results[
                case_name
            ] = evaluate_case(
                archive=ood_archive,
                case_name=case_name,
                bed_method=bed_method,
                bed_state=bed_state,
                wb_method=wb_method,
                wb_state=wb_state,
                sample_index=args.sample_index,
                steps_per_segment=(
                    args.steps_per_segment
                ),
            )

        # ========================================================
        # PLOT 1
        # ========================================================

        direct_case = args.direct_case

        if (
            direct_case
            not in gaussian_results
        ):

            raise ValueError(
                "--direct-case must be one of "
                f"{GAUSSIAN_CASES}"
            )

        print()
        print(
            "Creating Plot 1: "
            "Direct Condition Test..."
        )

        plot_direct_condition_test(
            data=gaussian_results[
                direct_case
            ][
                "data"
            ],
            bed_method=bed_method,
            bed_state=bed_state,
            wb_method=wb_method,
            wb_state=wb_state,
            plot_time=args.plot_time,
            output_path=(
                output_dir
                /
                "01_direct_condition_test.png"
            ),
        )

        # ========================================================
        # PLOT 2
        # ========================================================

        print(
            "Creating Plot 2: "
            "Isolated hill effect error..."
        )

        plot_isolated_hill_errors(
            gaussian_results=gaussian_results,
            output_path=(
                output_dir
                /
                "02_isolated_hill_effect_error.png"
            ),
            csv_path=(
                output_dir
                /
                "02_isolated_hill_effect_error.csv"
            ),
        )

        # ========================================================
        # PLOT 3
        # ========================================================

        print(
            "Creating Plot 3: "
            "Gaussian terrain-induced speed..."
        )

        plot_gaussian_speed(
            gaussian_results=gaussian_results,
            plot_time=args.plot_time,
            output_path=(
                output_dir
                /
                "03_terrain_induced_speed_gaussian.png"
            ),
        )

        # ========================================================
        # PLOT 4
        # ========================================================

        print(
            "Creating Plot 4: "
            "Unseen terrain-induced speed..."
        )

        plot_unseen_speed(
            ood_results=ood_results,
            plot_time=args.plot_time,
            output_path=(
                output_dir
                /
                "04_terrain_induced_speed_unseen.png"
            ),
        )

        # ========================================================
        # SAVE NUMERICAL ROLLOUTS
        # ========================================================

        save_dictionary = {}

        for case_name, result in (
            gaussian_results.items()
        ):

            prefix = (
                "gaussian_"
                +
                case_name
            )

            save_dictionary[
                prefix
                +
                "_pyclaw_hill"
            ] = result[
                "data"
            ][
                "terrain_q"
            ]

            save_dictionary[
                prefix
                +
                "_pyclaw_flat"
            ] = result[
                "data"
            ][
                "flat_q"
            ]

            save_dictionary[
                prefix
                +
                "_bathymetry"
            ] = result[
                "data"
            ][
                "b"
            ]

            save_dictionary[
                prefix
                +
                "_bed_hill"
            ] = result[
                "bed_hill"
            ]

            save_dictionary[
                prefix
                +
                "_bed_flat"
            ] = result[
                "bed_flat"
            ]

            save_dictionary[
                prefix
                +
                "_wb_hill"
            ] = result[
                "wb_hill"
            ]

            save_dictionary[
                prefix
                +
                "_wb_flat"
            ] = result[
                "wb_flat"
            ]

        for case_name, result in (
            ood_results.items()
        ):

            prefix = (
                "ood_"
                +
                case_name
            )

            save_dictionary[
                prefix
                +
                "_pyclaw_hill"
            ] = result[
                "data"
            ][
                "terrain_q"
            ]

            save_dictionary[
                prefix
                +
                "_pyclaw_flat"
            ] = result[
                "data"
            ][
                "flat_q"
            ]

            save_dictionary[
                prefix
                +
                "_bathymetry"
            ] = result[
                "data"
            ][
                "b"
            ]

            save_dictionary[
                prefix
                +
                "_bed_hill"
            ] = result[
                "bed_hill"
            ]

            save_dictionary[
                prefix
                +
                "_bed_flat"
            ] = result[
                "bed_flat"
            ]

            save_dictionary[
                prefix
                +
                "_wb_hill"
            ] = result[
                "wb_hill"
            ]

            save_dictionary[
                prefix
                +
                "_wb_flat"
            ] = result[
                "wb_flat"
            ]

        np.savez_compressed(
            output_dir
            /
            "diagnostic_rollouts.npz",
            **save_dictionary,
        )

        print()
        print("=" * 72)
        print("FINISHED")
        print("=" * 72)

        print(
            "Plots saved to:"
        )

        print(
            output_dir
        )

        print()

        print(
            "Files:"
        )

        print(
            "01_direct_condition_test.png"
        )

        print(
            "02_isolated_hill_effect_error.png"
        )

        print(
            "03_terrain_induced_speed_gaussian.png"
        )

        print(
            "04_terrain_induced_speed_unseen.png"
        )

        print(
            "02_isolated_hill_effect_error.csv"
        )

        print(
            "diagnostic_rollouts.npz"
        )

        print("=" * 72)

    finally:

        gaussian_archive.close()
        ood_archive.close()


if __name__ == "__main__":

    main()