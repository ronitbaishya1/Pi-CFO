"""
Run controlled 64x64 lambda sweeps for the final Geometry-U-FNO
WB-Bed-PI-CFO model.

The underlying training implementation is NOT modified.

Default screening candidates:

    (lambda_PDE, lambda_bed, lambda_WB)

    (0.01, 0.70, 0.10)
    (0.01, 0.40, 0.10)
    (0.02, 0.40, 0.10)

These are trained from scratch using the same:

    dataset
    architecture
    optimizer
    seed
    training length
    resolution

Optionally, the script can also run the full existing 01-08
diagnostic suite for every candidate.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from pathlib import Path


# =====================================================================
# PROJECT ROOT
# =====================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Controlled 64x64 Geometry-U-FNO "
            "physics-loss sweep."
        )
    )

    # -----------------------------------------------------------------
    # DATA
    # -----------------------------------------------------------------

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_64_id.h5"
        ),
    )

    # Format:
    #
    # "pde,bed,wb;pde,bed,wb;..."
    #
    parser.add_argument(
        "--candidates",
        type=str,
        default=(
            "0.01,0.70,0.10;"
            "0.01,0.40,0.10;"
            "0.02,0.40,0.10"
        ),
    )

    # -----------------------------------------------------------------
    # TRAINING
    # -----------------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--spline-batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1.0e-4,
    )

    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--beta2",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0e-5,
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=2,
    )

    # -----------------------------------------------------------------
    # FINAL GEOMETRY-U-FNO ARCHITECTURE
    # -----------------------------------------------------------------

    parser.add_argument(
        "--modes1",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--modes2",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-blocks",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-u-blocks",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--geometry-width",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--geometry-depth",
        type=int,
        default=2,
    )

    # -----------------------------------------------------------------
    # PHYSICS
    # -----------------------------------------------------------------

    parser.add_argument(
        "--wb-eta0",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--gravity",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--x-length",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--y-length",
        type=float,
        default=5.0,
    )

    # -----------------------------------------------------------------
    # OUTPUT ROOTS
    # -----------------------------------------------------------------

    parser.add_argument(
        "--study-name",
        type=str,
        default="screen3",
    )

    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=(
            "checkpoints/"
            "lambda_sweep_64"
        ),
    )

    parser.add_argument(
        "--results-root",
        type=str,
        default=(
            "results/"
            "lambda_sweep_64"
        ),
    )

    parser.add_argument(
        "--force-train",
        action="store_true",
    )

    # -----------------------------------------------------------------
    # OPTIONAL FULL DIAGNOSTIC SUITE
    # -----------------------------------------------------------------

    parser.add_argument(
        "--run-diagnostics",
        action="store_true",
    )

    parser.add_argument(
        "--force-diagnostics",
        action="store_true",
    )

    parser.add_argument(
        "--counterfactual-data",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "gaussian_counterfactual_64.h5"
        ),
    )

    parser.add_argument(
        "--ood-data",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_ood_64.h5"
        ),
    )

    parser.add_argument(
        "--cfo-ckpt-dir",
        type=str,
        default=(
            "checkpoints/"
            "final_resolution_study/"
            "bathy_cfo/"
            "res64/"
            "seed0/"
            "best"
        ),
    )

    parser.add_argument(
        "--cfo-prefix",
        type=str,
        default="bathy_cfo",
    )

    parser.add_argument(
        "--plot-time",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--direct-case",
        type=str,
        default="hill_right",
    )

    parser.add_argument(
        "--cross-section-case",
        type=str,
        default="hill_center",
    )

    return parser.parse_args()


# =====================================================================
# PATH
# =====================================================================

def resolve_path(
    value,
):

    path = (
        Path(value)
        .expanduser()
    )

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            /
            path
        )

    return path.resolve()


# =====================================================================
# LAMBDA STRING
# =====================================================================

def lambda_text(
    value,
):

    text = (
        f"{float(value):.6g}"
    )

    text = (
        text
        .replace(
            "-",
            "m",
        )
        .replace(
            ".",
            "p",
        )
    )

    return text


# =====================================================================
# RUN TAG
# =====================================================================

def run_tag(
    lambda_pde,
    lambda_bed,
    lambda_wb,
):

    return (
        "pde_"
        +
        lambda_text(
            lambda_pde
        )
        +
        "__bed_"
        +
        lambda_text(
            lambda_bed
        )
        +
        "__wb_"
        +
        lambda_text(
            lambda_wb
        )
    )


# =====================================================================
# CANDIDATES
# =====================================================================

def parse_candidates(
    text,
):

    candidates = []

    blocks = [
        block.strip()
        for block
        in text.split(";")
        if block.strip()
    ]

    for block in blocks:

        values = [
            value.strip()
            for value
            in block.split(",")
        ]

        if len(values) != 3:

            raise ValueError(
                "Each candidate must contain exactly "
                "three comma-separated values:\n"
                "lambda_PDE,lambda_bed,lambda_WB\n\n"
                f"Invalid candidate: {block}"
            )

        candidates.append(
            (
                float(
                    values[0]
                ),
                float(
                    values[1]
                ),
                float(
                    values[2]
                ),
            )
        )

    if not candidates:

        raise ValueError(
            "No lambda candidates supplied."
        )

    return candidates


# =====================================================================
# RUN COMMAND
# =====================================================================

def run_command(
    command,
):

    print()
    print(
        "=" * 80
    )

    print(
        "RUNNING"
    )

    print(
        "=" * 80
    )

    print(
        " ".join(
            str(item)
            for item
            in command
        )
    )

    print(
        "=" * 80
    )

    subprocess.run(
        command,
        check=True,
        cwd=str(
            PROJECT_ROOT
        ),
    )


# =====================================================================
# TRAIN ONE CANDIDATE
# =====================================================================

def train_candidate(
    args,
    *,
    lambda_pde,
    lambda_bed,
    lambda_wb,
    checkpoint_dir,
    results_dir,
):

    training_script = (
        PROJECT_ROOT
        /
        "scripts"
        /
        "train_geometry_wb_bathy_bed_pi_cfo.py"
    )

    command = [
        sys.executable,
        str(
            training_script
        ),

        "--dataset-path",
        str(
            resolve_path(
                args.dataset_path
            )
        ),

        "--architecture",
        "geometry_ufno",

        "--modes1",
        str(
            args.modes1
        ),

        "--modes2",
        str(
            args.modes2
        ),

        "--width",
        str(
            args.width
        ),

        "--num-blocks",
        str(
            args.num_blocks
        ),

        "--num-u-blocks",
        str(
            args.num_u_blocks
        ),

        "--geometry-width",
        str(
            args.geometry_width
        ),

        "--geometry-depth",
        str(
            args.geometry_depth
        ),

        "--epochs",
        str(
            args.epochs
        ),

        "--batch-size",
        str(
            args.batch_size
        ),

        "--spline-batch-size",
        str(
            args.spline_batch_size
        ),

        "--seed",
        str(
            args.seed
        ),

        "--lr",
        str(
            args.lr
        ),

        "--beta1",
        str(
            args.beta1
        ),

        "--beta2",
        str(
            args.beta2
        ),

        "--gamma",
        str(
            args.gamma
        ),

        "--lambda-pde",
        str(
            lambda_pde
        ),

        "--lambda-bed",
        str(
            lambda_bed
        ),

        "--lambda-wb",
        str(
            lambda_wb
        ),

        "--wb-eta0",
        str(
            args.wb_eta0
        ),

        "--gravity",
        str(
            args.gravity
        ),

        "--x-length",
        str(
            args.x_length
        ),

        "--y-length",
        str(
            args.y_length
        ),

        "--eval-interval",
        str(
            args.eval_interval
        ),

        "--steps-per-segment",
        str(
            args.steps_per_segment
        ),

        "--ckpt-dir",
        str(
            checkpoint_dir
        ),

        "--results-dir",
        str(
            results_dir
        ),
    ]

    run_command(
        command
    )


# =====================================================================
# FULL DIAGNOSTIC SUITE
# =====================================================================

def run_diagnostics(
    args,
    *,
    lambda_pde,
    lambda_bed,
    lambda_wb,
    checkpoint_dir,
    diagnostics_dir,
):

    suite_script = (
        PROJECT_ROOT
        /
        "experiments"
        /
        "run_final_resolution_suite.py"
    )

    if not suite_script.exists():

        raise FileNotFoundError(
            "The resolution-suite wrapper was not found:\n"
            f"{suite_script}\n\n"
            "Create run_final_resolution_suite.py first."
        )

    command = [
        sys.executable,
        str(
            suite_script
        ),

        "--resolution",
        "64",

        "--source-resolution",
        "64",

        "--id-data",
        str(
            resolve_path(
                args.dataset_path
            )
        ),

        "--counterfactual-data",
        str(
            resolve_path(
                args.counterfactual_data
            )
        ),

        "--ood-data",
        str(
            resolve_path(
                args.ood_data
            )
        ),

        "--cfo-ckpt-dir",
        str(
            resolve_path(
                args.cfo_ckpt_dir
            )
        ),

        "--cfo-prefix",
        args.cfo_prefix,

        "--final-ckpt",
        str(
            checkpoint_dir
            /
            "best"
        ),

        "--lambda-pde",
        str(
            lambda_pde
        ),

        "--lambda-bed",
        str(
            lambda_bed
        ),

        "--lambda-wb",
        str(
            lambda_wb
        ),

        "--wb-eta0",
        str(
            args.wb_eta0
        ),

        "--plot-time",
        str(
            args.plot_time
        ),

        "--sample-index",
        "0",

        "--direct-case",
        args.direct_case,

        "--cross-section-case",
        args.cross_section_case,

        "--steps-per-segment",
        str(
            args.steps_per_segment
        ),

        "--output-dir",
        str(
            diagnostics_dir
        ),
    ]

    run_command(
        command
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    candidates = parse_candidates(
        args.candidates
    )

    checkpoint_root = (
        resolve_path(
            args.checkpoint_root
        )
        /
        args.study_name
    )

    results_root = (
        resolve_path(
            args.results_root
        )
        /
        args.study_name
    )

    checkpoint_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 80
    )

    print(
        "64x64 GEOMETRY-U-FNO LOSS SWEEP"
    )

    print(
        "=" * 80
    )

    print(
        "Dataset:",
        resolve_path(
            args.dataset_path
        ),
    )

    print(
        "Epochs per run:",
        args.epochs,
    )

    print(
        "Seed:",
        args.seed,
    )

    print(
        "Fourier modes:",
        (
            args.modes1,
            args.modes2,
        ),
    )

    print(
        "Number of candidates:",
        len(
            candidates
        ),
    )

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"  {index}. "
            f"PDE={candidate[0]}, "
            f"bed={candidate[1]}, "
            f"WB={candidate[2]}"
        )

    print(
        "=" * 80
    )

    for (
        lambda_pde,
        lambda_bed,
        lambda_wb,
    ) in candidates:

        tag = run_tag(
            lambda_pde,
            lambda_bed,
            lambda_wb,
        )

        checkpoint_dir = (
            checkpoint_root
            /
            tag
            /
            f"seed{args.seed}"
        )

        results_dir = (
            results_root
            /
            tag
            /
            f"seed{args.seed}"
        )

        diagnostics_dir = (
            results_dir
            /
            "diagnostics"
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        metrics_path = (
            results_dir
            /
            "metrics.npz"
        )

        # ---------------------------------------------------------
        # TRAIN
        # ---------------------------------------------------------

        if (
            metrics_path.exists()
            and
            not args.force_train
        ):

            print()
            print(
                f"[skip training] {tag}"
            )

            print(
                "Existing metrics:",
                metrics_path,
            )

        else:

            train_candidate(
                args,

                lambda_pde=(
                    lambda_pde
                ),

                lambda_bed=(
                    lambda_bed
                ),

                lambda_wb=(
                    lambda_wb
                ),

                checkpoint_dir=(
                    checkpoint_dir
                ),

                results_dir=(
                    results_dir
                ),
            )

        # ---------------------------------------------------------
        # FULL 01-08 DIAGNOSTICS
        # ---------------------------------------------------------

        if args.run_diagnostics:

            final_summary = (
                diagnostics_dir
                /
                "tables"
                /
                "table_08_final_summary.csv"
            )

            if (
                final_summary.exists()
                and
                not args.force_diagnostics
            ):

                print()
                print(
                    f"[skip diagnostics] {tag}"
                )

                print(
                    "Existing summary:",
                    final_summary,
                )

            else:

                run_diagnostics(
                    args,

                    lambda_pde=(
                        lambda_pde
                    ),

                    lambda_bed=(
                        lambda_bed
                    ),

                    lambda_wb=(
                        lambda_wb
                    ),

                    checkpoint_dir=(
                        checkpoint_dir
                    ),

                    diagnostics_dir=(
                        diagnostics_dir
                    ),
                )

    print()
    print(
        "=" * 80
    )

    print(
        "SWEEP COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "Checkpoints:",
        checkpoint_root,
    )

    print(
        "Results:",
        results_root,
    )


if __name__ == "__main__":

    main()