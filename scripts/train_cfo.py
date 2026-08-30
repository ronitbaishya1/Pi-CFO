"""CLI entrypoint for CFO training experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.readers import load_dataset_splits
from cfo import ContinuousFlowOperator
from models.factory import build_model
from train import CFOTrainArgs, train_cfo
from utils.data import build_dataloader, linear_spline, quintic_spline_batch, load_partial_data
from utils.metrics import relative_L2_error, relative_frobenius_error, rmse
from utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CFO model")

    # run setup
    parser.add_argument("--epochs", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)

    # optimization / CFO dynamics
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--gamma", type=float, default=1e-5)

    # data / model
    parser.add_argument("--dataset", type=str, default="lorenz", choices=["lorenz", "burgers", "dr", "swe"])
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--model", type=str, default="SimpleMLP", choices=["UNet1D", "UNet2D", "FNO1d", "FNO2d", "DiT", "SimpleMLP"])
    parser.add_argument("--spline-type", type=str, default="quintic", choices=["linear", "quintic"])
    parser.add_argument("--spline-batch-size", type=int, default=32, help="Batch size used only for spline coefficient construction")
    parser.add_argument("--partial-train-ratio", type=float, default=1.0, help="Fraction of training time snapshots used to build splines")

    # evaluation / checkpointing
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt-prefix", type=str, default="cfo_lorenz")
    parser.add_argument("--running-ckpt-interval", type=int, default=None, help="Save running checkpoint every N epochs; defaults to eval-interval")
    parser.add_argument("--running-ckpt-max-to-keep", type=int, default=3, help="Number of running checkpoints to keep")

    # logging / utility
    parser.add_argument("--no-eval", action="store_true", help="Disable periodic evaluation during training")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    return parser.parse_args()


def _build_spline_dataset(train_data, spline_type: str, batch_size: int, spline_batch_size: int, epochs: int, seed: int, time=None):
    batch_n, traj_len = train_data.shape[:2]
    if time is None:
        time = np.broadcast_to(np.linspace(0.0, 1.0, traj_len, dtype=train_data.dtype), (batch_n, traj_len))

    if spline_type == "linear":
        spline_coef, start_time, end_time = linear_spline(train_data, time=time)
    else:
        spline_coef, start_time, end_time = quintic_spline_batch(train_data, time=time, batch_size=spline_batch_size)

    return build_dataloader(
        spline_coef,
        t1=start_time,
        t2=end_time,
        batch_size=batch_size,
        num_epochs=epochs,
        seed=seed,
    )

def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    use_condition = False
    effective_lr = float(args.lr)
    effective_beta1 = float(args.beta1)
    effective_beta2 = float(args.beta2)

    if args.dry_run:
        print(args)
        return

    splits = load_dataset_splits(args.dataset, file_path=args.dataset_path)
    train_data = splits["train"]
    eval_data = splits["eval"]
    test_data = splits["test"]

    train_time = None
    if args.partial_train_ratio < 1.0:
        train_data, train_time = load_partial_data(train_data, ratio=args.partial_train_ratio, seed=args.seed)

    input_shape = tuple(train_data.shape[2:])
    model = build_model(args.model, input_shape, use_condition=use_condition)

    task_desc = "partial snapshots" if args.partial_train_ratio < 1.0 else "full trajectories"
    print("=== TASK SUMMARY ===")
    print(
        f"{args.spline_type.capitalize()} CFO learn with {args.model} from ratio={args.partial_train_ratio:.3f} "
        f"({task_desc}) observation of {args.dataset} dataset."
    )

    print("=== CFO Training Run ===")
    print(f"dataset={args.dataset} model={args.model} spline_type={args.spline_type}")
    print(f"seed={args.seed} epochs={args.epochs} batch_size={args.batch_size} spline_batch_size={args.spline_batch_size}")
    print(f"lr={effective_lr} beta1={effective_beta1} beta2={effective_beta2}")
    print(f"eval_interval={args.eval_interval} do_eval={not args.no_eval}")
    print(f"train/eval/test sizes = {len(train_data)}/{len(eval_data)}/{len(test_data)}")
    print(f"input_shape={input_shape}")
    running_ckpt_interval = args.eval_interval if args.running_ckpt_interval is None else args.running_ckpt_interval
    print(f"checkpoint_policy running_every={running_ckpt_interval} keep={args.running_ckpt_max_to_keep} best_keep=1")

    print("Preparing spline dataloader...")
    
    spline_loader = _build_spline_dataset(
        train_data=train_data,
        spline_type=args.spline_type,
        batch_size=args.batch_size,
        spline_batch_size=args.spline_batch_size,
        epochs=args.epochs,
        seed=args.seed,
        time=train_time,
    )
    print("Spline construction complete. Starting training...")

    method = ContinuousFlowOperator(
        model=model,
        input_shape=input_shape,
        gamma=float(args.gamma),
        spline_type=args.spline_type,
        use_condition=use_condition,
    )
    train_args = CFOTrainArgs(
        num_epochs=args.epochs,
        random_seed=args.seed,
        use_wandb=args.use_wandb,
        learning_rate=effective_lr,
        beta1=effective_beta1,
        beta2=effective_beta2,
        do_eval=not args.no_eval,
        eval_interval=args.eval_interval,
        irregular_time=False,
        running_ckpt_dir=str((Path(args.ckpt_dir).resolve() / "running")),
        running_ckpt_prefix=args.ckpt_prefix,
        running_ckpt_interval=running_ckpt_interval,
        running_ckpt_max_to_keep=args.running_ckpt_max_to_keep,
        best_ckpt_dir=str((Path(args.ckpt_dir).resolve() / "best")),
        best_ckpt_prefix=args.ckpt_prefix,
    ) 

    eval_dataset = (eval_data[:, 0], eval_data)
    print("Starting training loop...")
    train_output = train_cfo(method, spline_loader, train_args, eval_dataset=eval_dataset)

    state_for_test = train_output["state"]
    if np.isfinite(train_output["best_l2_error"]):
        state_for_test = train_output["best_state"]

    print("Running final test inference...")
    test_pred = method.uniform_inference(
        state_for_test,
        test_data[:, 0],
        trajectory_points_num=test_data.shape[1],
        steps_per_segment=2,
        method="RK4",
    )
    test_rmse = rmse(test_data, test_pred)
    test_rel_l2 = relative_L2_error(test_data, test_pred)
    test_rel_fro = relative_frobenius_error(test_data, test_pred)

    best_epoch = train_output["best_epoch"]
    best_rel_l2 = train_output["best_l2_error"]

    if np.isfinite(best_rel_l2):
        print(f"Best eval checkpoint: epoch={best_epoch} rel_l2={best_rel_l2:.6f}")
    else:
        print("Best eval checkpoint: none (evaluation disabled or not run)")

    print("Final held-out test metrics:")
    print(f"  Relative L2 Error: {float(test_rel_l2):.6f}")
    print(f"  RMSE: {float(test_rmse):.6f}")
    print(f"  Relative Frobenius Error: {float(test_rel_fro):.6f}")

    ckpt_dir = Path(args.ckpt_dir).resolve()
    print(f"Running checkpoints: dir={ckpt_dir / 'running'} keep={args.running_ckpt_max_to_keep}")
    print(f"Best checkpoint: dir={ckpt_dir / 'best'} keep=1")


if __name__ == "__main__":
    main()


