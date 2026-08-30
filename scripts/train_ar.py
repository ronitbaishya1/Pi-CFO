"""CLI entrypoint for autoregressive baseline training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.readers import load_dataset_splits
from autoregressive import Autoregressive
from models.factory import build_model
from train import ARTrainArgs, train_ar
from utils.data import autoregressive_dataset, build_dataloader
from utils.metrics import relative_L2_error, relative_frobenius_error, rmse
from utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AR baseline")

    # run setup
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)

    # optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)

    # data/model
    parser.add_argument("--dataset", type=str, default="lorenz", choices=["lorenz", "burgers", "dr", "swe"])
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--model", type=str, default="UNet1D", choices=["UNet1D", "UNet2D", "FNO1d", "FNO2d", "DiT", "SimpleMLP"])
    parser.add_argument("--use-time", action="store_true", help="Use time embedding in AR model")

    # evaluation/checkpointing
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt-prefix", type=str, default="ar_")
    parser.add_argument("--running-ckpt-interval", type=int, default=None, help="Save running checkpoint every N epochs; defaults to eval-interval")
    parser.add_argument("--running-ckpt-max-to-keep", type=int, default=3, help="Number of running checkpoints to keep")

    # logging/utility
    parser.add_argument("--no-eval", action="store_true", help="Disable periodic evaluation during training")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
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
    input_shape = tuple(train_data.shape[2:])
    trajectory_points_num = int(train_data.shape[1])
    model = build_model(args.model, input_shape, use_condition=False)

    x, y = autoregressive_dataset(train_data)
    dataloader = build_dataloader(
        X=x,
        y=y,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        seed=args.seed,
    )

    method = Autoregressive(
        model=model,
        input_shape=input_shape,
        trajectory_points_num=trajectory_points_num,
        use_time=bool(args.use_time),
    )
    running_ckpt_interval = args.eval_interval if args.running_ckpt_interval is None else args.running_ckpt_interval
    train_args = ARTrainArgs(
        num_epochs=args.epochs,
        random_seed=args.seed,
        use_wandb=args.use_wandb,
        learning_rate=effective_lr,
        beta1=effective_beta1,
        beta2=effective_beta2,
        do_eval=not args.no_eval,
        eval_interval=args.eval_interval,
        running_ckpt_dir=str((Path(args.ckpt_dir).resolve() / "running")),
        running_ckpt_prefix=args.ckpt_prefix,
        running_ckpt_interval=running_ckpt_interval,
        running_ckpt_max_to_keep=args.running_ckpt_max_to_keep,
        best_ckpt_dir=str((Path(args.ckpt_dir).resolve() / "best")),
        best_ckpt_prefix=args.ckpt_prefix,
    )
    eval_dataset = (eval_data[:, 0], eval_data)
    train_output = train_ar(method, dataloader, train_args, eval_dataset=eval_dataset)

    state_for_test = train_output["state"]
    if train_output["best_l2_error"] < float("inf"):
        state_for_test = train_output["best_state"]

    test_pred = method.inference(state_for_test, test_data[:, 0])
    test_rmse = rmse(test_data, test_pred)
    test_rel_l2 = relative_L2_error(test_data, test_pred)
    test_rel_fro = relative_frobenius_error(test_data, test_pred)

    best_epoch = train_output["best_epoch"]
    best_rel_l2 = train_output["best_l2_error"]
    if best_rel_l2 < float("inf"):
        print(f"Best eval checkpoint: epoch={best_epoch} rel_l2={best_rel_l2:.6f}")
    else:
        print("Best eval checkpoint: none (evaluation disabled or not run)")

    print("Final held-out test metrics:")
    print(f"  RMSE: {float(test_rmse):.6f}")
    print(f"  Relative L2 Error: {float(test_rel_l2):.6f}")
    print(f"  Relative Frobenius Error: {float(test_rel_fro):.6f}")

    ckpt_dir = Path(args.ckpt_dir).resolve()
    print(f"Running checkpoints: dir={ckpt_dir / 'running'} keep={args.running_ckpt_max_to_keep}")
    print(f"Best checkpoint: dir={ckpt_dir / 'best'} keep=1")


if __name__ == "__main__":
    main()
