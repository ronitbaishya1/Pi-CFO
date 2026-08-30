"""CLI entrypoint for CFO checkpoint evaluation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cfo import ContinuousFlowOperator
from models.factory import build_model
from utils.readers import load_dataset_splits
from train import init_cfo_train_state
from utils.checkpoints import load_train_state
from utils.data import select_data_split
from utils.metrics import relative_frobenius_error, relative_L2_error, rmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved CFO checkpoint on a dataset split")

    # data / model
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["lorenz", "burgers", "dr", "swe"],
        help="Dataset name",
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="Optional dataset file path override")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["UNet1D", "UNet2D", "FNO1d", "FNO2d", "DiT", "SimpleMLP"],
        help="Model architecture used for the checkpoint",
    )

    # checkpoint restore
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Checkpoint directory (e.g., checkpoints/best)")
    parser.add_argument("--ckpt-prefix", type=str, required=True, help="Checkpoint prefix (run name)")
    parser.add_argument("--ckpt-step", type=int, default=None, help="Specific checkpoint step (default: latest)")

    # evaluation control
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "eval", "test"],
        help="Dataset split used for evaluation",
    )
    parser.add_argument("--steps-per-segment", type=int, default=2, help="CFO rollout steps per segment")
    parser.add_argument("--solver", type=str, default="RK4", choices=["RK4", "Euler", "Heun"], help="CFO ODE solver")

    # output
    parser.add_argument("--save-pred", type=str, default=None, help="Optional path to save prediction .npy")
    return parser.parse_args()


def _load_target(args: argparse.Namespace) -> np.ndarray:
    splits = load_dataset_splits(args.dataset, file_path=args.dataset_path)
    return select_data_split(splits, args.split)


def _restore_state(args: argparse.Namespace, target: np.ndarray):
    input_shape = tuple(target.shape[2:])
    model = build_model(args.model, input_shape, use_condition=False)

    method = ContinuousFlowOperator(
        model=model,
        input_shape=input_shape,
        spline_type="quintic",
        use_condition=False,
    )
    state_template = init_cfo_train_state(
        method,
        seed=0,
    )

    ckpt_dir = str(Path(args.ckpt_dir).resolve())
    state = load_train_state(
        target_state=state_template,
        ckpt_dir=ckpt_dir,
        prefix=args.ckpt_prefix,
        step=args.ckpt_step,
    )
    return method, state, ckpt_dir


def main() -> None:
    args = parse_args()

    target = _load_target(args)
    method, state, ckpt_dir = _restore_state(args, target)

    pred = method.uniform_inference(
        state,
        target[:, 0],
        trajectory_points_num=target.shape[1],
        steps_per_segment=args.steps_per_segment,
        method=args.solver,
    )

    if args.save_pred:
        np.save(args.save_pred, pred)

    ckpt_step_desc = args.ckpt_step if args.ckpt_step is not None else "latest"
    print(f"Target shape: {target.shape}")
    print(f"Pred shape:   {pred.shape}")
    print(f"Method: cfo")
    print(f"Target source: dataset={args.dataset} split={args.split}")
    print(f"Inference: solver={args.solver} steps_per_segment={args.steps_per_segment}")
    print(f"Checkpoint: dir={ckpt_dir} prefix={args.ckpt_prefix} step={ckpt_step_desc}")
    print(f"Relative L2: {relative_L2_error(target, pred):.6f}")
    print(f"RMSE: {rmse(target, pred):.6f}")
    print(f"Relative Frobenius: {relative_frobenius_error(target, pred):.6f}")
    if args.save_pred:
        print(f"Saved prediction array: {Path(args.save_pred).resolve()}")


if __name__ == "__main__":
    main()
