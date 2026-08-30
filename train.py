"""Project-level train entry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import value_and_grad
from jax.tree_util import tree_leaves
from tqdm.auto import trange

from utils.data import prepare_tf_data, prefetch_to_device
from utils.checkpoints import save_train_state
from utils.logging import ExperimentLogger
from utils.metrics import relative_L2_error, relative_frobenius_error, rmse




@dataclass
class CFOTrainArgs:
    num_epochs: int = 1000
    random_seed: int = 0
    use_wandb: bool = False
    log_mode: str = "auto"
    train_log_interval: int = 1
    console_logging: bool = True
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    do_eval: bool = False
    eval_interval: int = 500
    irregular_time: bool = False
    running_ckpt_dir: str | None = None
    running_ckpt_prefix: str = "running_"
    running_ckpt_interval: int = 0
    running_ckpt_max_to_keep: int = 3
    best_ckpt_dir: str | None = None
    best_ckpt_prefix: str = "best_"


@dataclass
class ARTrainArgs:
    num_epochs: int = 1000
    random_seed: int = 0
    use_wandb: bool = False
    log_mode: str = "auto"
    train_log_interval: int = 1
    console_logging: bool = True
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    do_eval: bool = False
    eval_interval: int = 500
    running_ckpt_dir: str | None = None
    running_ckpt_prefix: str = "running_"
    running_ckpt_interval: int = 0
    running_ckpt_max_to_keep: int = 3
    best_ckpt_dir: str | None = None
    best_ckpt_prefix: str = "best_"


def _prepare_cfo_train_batch(method, raw_batch, time_key, noise_key):
    if method.use_condition:
        spline_coef, t_start, t_end, condition, *_ = raw_batch
    else:
        spline_coef, t_start, t_end, *_ = raw_batch
        condition = None

    spline_coef = spline_coef[0]
    t_start = t_start[0]
    t_end = t_end[0]
    dt = t_end - t_start
    delta_t = jax.random.uniform(time_key, shape=(len(dt),), minval=0.0, maxval=dt)

    x0 = spline_coef[:, 0]
    eps = jax.random.normal(noise_key, x0.shape)

    if method.use_condition:
        condition = condition[0]
        return (spline_coef, condition, t_start, t_end, delta_t, eps)
    return (spline_coef, t_start, t_end, delta_t, eps)


def _run_cfo_eval(method, state, epoch: int, eval_dataset, logger: ExperimentLogger):
    if method.use_condition:
        x0_eval, target_eval, condition_eval = eval_dataset
    else:
        x0_eval, target_eval = eval_dataset
        condition_eval = None

    pred_eval = method.uniform_inference(
        state,
        x0_eval,
        trajectory_points_num=target_eval.shape[1],
        steps_per_segment=2,
        condition=condition_eval,
        method="RK4",
    )
    rel_fro_value = relative_frobenius_error(target_eval, pred_eval)
    rmse_value = rmse(target_eval, pred_eval)
    rel_l2_value = relative_L2_error(target_eval, pred_eval)

    logger.log(
        {
            "eval/epoch": epoch,
            "eval/Relative_L2_Error": rel_l2_value,
            "eval/RMSE": rmse_value,
            "eval/Relative_Frobenius_Error": rel_fro_value,
        },
        commit=False,
    )
    return float(rel_l2_value), float(rmse_value), float(rel_fro_value)


def init_cfo_train_state(
    method,
    *,
    seed: int,
    learning_rate: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.99,
) -> TrainState:
    rng_key = jax.random.PRNGKey(seed)
    x = jnp.ones((1,) + tuple(method.input_shape), dtype=jnp.float32)
    t = jnp.ones((1,), dtype=jnp.float32)
    if method.use_condition:
        c = jnp.ones((1,) + tuple(method.condition_shape), dtype=jnp.float32)
        variables = method.model.init(rng_key, x, t, c)
    else:
        variables = method.model.init(rng_key, x, t)

    tx = optax.adam(learning_rate=float(learning_rate), b1=float(beta1), b2=float(beta2))
    return TrainState.create(apply_fn=method.model.apply, params=variables['params'], tx=tx)


def train_cfo(method, spline_dataloader, args: CFOTrainArgs, eval_dataset: Optional[Tuple[np.ndarray, np.ndarray]] = None):
    state = init_cfo_train_state(
        method,
        seed=args.random_seed,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
    )
    loss_fn = method.loss_fn

    def cfo_train_step(state: TrainState, batch):
        loss, grads = value_and_grad(loss_fn)(state.params, batch)
        state = state.apply_gradients(grads=grads)
        return loss, state

    step_jit = jax.jit(cfo_train_step)

    logger = ExperimentLogger.create(
        use_wandb=args.use_wandb,
        log_mode=args.log_mode,
        train_log_interval=args.train_log_interval,
        console=args.console_logging,
    )
    num_params = sum(x.size for x in tree_leaves(state.params))
    logger.info(f"Model parameters: {int(num_params)}")

    rng_key = jax.random.PRNGKey(args.random_seed)
    pbar = trange(args.num_epochs, desc="Training")
    data = map(prepare_tf_data, spline_dataloader)
    data = prefetch_to_device(data, 2)

    loss_log: list[float] = []
    best_state = state
    best_l2_error = float("inf")
    best_epoch = -1

    for epoch in pbar:
        rng_key, time_key, noise_key = jax.random.split(rng_key, 3)
        batch = _prepare_cfo_train_batch(method, next(data), time_key, noise_key)
        loss, state = step_jit(state, batch)

        should_save_running_interval = (
            args.running_ckpt_dir is not None
            and args.running_ckpt_interval > 0
            and (epoch + 1) % args.running_ckpt_interval == 0
        )

        if args.do_eval and (epoch % args.eval_interval == 0) and epoch > 0 and eval_dataset is not None:
            rel_l2_value, rmse_value, rel_fro_value = _run_cfo_eval(method, state, epoch, eval_dataset, logger)
            should_save_running_interval = should_save_running_interval or (args.running_ckpt_dir is not None)
            if rel_l2_value < best_l2_error:
                best_l2_error = rel_l2_value
                best_state = state
                best_epoch = epoch
                if args.best_ckpt_dir is not None:
                    save_train_state(
                        best_state,
                        args.best_ckpt_dir,
                        prefix=args.best_ckpt_prefix,
                        step=epoch,
                        max_to_keep=1,
                    )
                logger.info(
                    f"[eval] epoch={epoch} rel_l2={rel_l2_value:.6f} rmse={rmse_value:.6f} rel_fro={rel_fro_value:.6f} [BEST]"
                )
            else:
                logger.info(
                    f"[eval] epoch={epoch} rel_l2={rel_l2_value:.6f} rmse={rmse_value:.6f} rel_fro={rel_fro_value:.6f}"
                )

        if should_save_running_interval:
            save_train_state(
                state,
                args.running_ckpt_dir,
                prefix=args.running_ckpt_prefix,
                step=epoch,
                max_to_keep=args.running_ckpt_max_to_keep,
            )

        loss_value = float(loss)
        loss_log.append(loss_value)
        pbar.set_postfix({"loss": loss_value})
    print("Training complete.")

    return {
        "state": state,
        "best_state": best_state,
        "best_l2_error": best_l2_error,
        "best_epoch": best_epoch,
        "loss_log": loss_log,
    }


def _prepare_ar_train_batch(method, raw_batch):
    if method.use_time:
        x, y, t, *_ = raw_batch
        return (x[0], y[0], t[0])
    x, y, *_ = raw_batch
    return (x[0], y[0])


def _run_ar_eval(method, state, epoch: int, eval_dataset, logger: ExperimentLogger):
    x0_eval, target_eval = eval_dataset
    pred_eval = method.inference(state, x0_eval)
    rel_fro_value = relative_frobenius_error(target_eval, pred_eval)
    rmse_value = rmse(target_eval, pred_eval)
    rel_l2_value = relative_L2_error(target_eval, pred_eval)

    logger.log(
        {
            "eval/epoch": epoch,
            "eval/Relative_L2_Error": rel_l2_value,
            "eval/RMSE": rmse_value,
            "eval/Relative_Frobenius_Error": rel_fro_value,
        },
        commit=False,
    )
    return float(rel_l2_value), float(rmse_value), float(rel_fro_value)


def init_ar_train_state(
    method,
    *,
    seed: int,
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.99,
) -> TrainState:
    rng_key = jax.random.PRNGKey(seed)
    x = jnp.ones((1,) + tuple(method.input_shape), dtype=jnp.float32)
    if method.use_time:
        t = jnp.ones((1,), dtype=jnp.float32)
        variables = method.model.init(rng_key, x, t)
    else:
        variables = method.model.init(rng_key, x)

    tx = optax.adam(learning_rate=float(learning_rate), b1=float(beta1), b2=float(beta2))
    return TrainState.create(apply_fn=method.model.apply, params=variables['params'], tx=tx)


def train_ar(method, dataloader, args: ARTrainArgs, eval_dataset: Optional[Tuple[np.ndarray, np.ndarray]] = None):
    state = init_ar_train_state(
        method,
        seed=args.random_seed,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
    )
    loss_fn = method.loss_fn

    def ar_train_step(state: TrainState, batch):
        loss, grads = value_and_grad(loss_fn)(state.params, batch)
        state = state.apply_gradients(grads=grads)
        return loss, state

    step_jit = jax.jit(ar_train_step)

    logger = ExperimentLogger.create(
        use_wandb=args.use_wandb,
        log_mode=args.log_mode,
        train_log_interval=args.train_log_interval,
        console=args.console_logging,
    )
    num_params = sum(x.size for x in tree_leaves(state.params))
    logger.info(f"Model parameters: {int(num_params)}")

    pbar = trange(args.num_epochs, desc="Training")
    data = map(prepare_tf_data, dataloader)
    data = prefetch_to_device(data, 2)

    loss_log: list[float] = []
    best_state = state
    best_l2_error = float("inf")
    best_epoch = -1

    for epoch in pbar:
        batch = _prepare_ar_train_batch(method, next(data))
        loss, state = step_jit(state, batch)

        should_save_running_interval = (
            args.running_ckpt_dir is not None
            and args.running_ckpt_interval > 0
            and (epoch + 1) % args.running_ckpt_interval == 0
        )

        if args.do_eval and (epoch % args.eval_interval == 0) and epoch > 0 and eval_dataset is not None:
            rel_l2_value, rmse_value, rel_fro_value = _run_ar_eval(method, state, epoch, eval_dataset, logger)
            should_save_running_interval = should_save_running_interval or (args.running_ckpt_dir is not None)
            if rel_l2_value < best_l2_error:
                best_l2_error = rel_l2_value
                best_state = state
                best_epoch = epoch
                if args.best_ckpt_dir is not None:
                    save_train_state(
                        best_state,
                        args.best_ckpt_dir,
                        prefix=args.best_ckpt_prefix,
                        step=epoch,
                        max_to_keep=1,
                    )
                logger.info(
                    f"[eval] epoch={epoch} rel_l2={rel_l2_value:.6f} rmse={rmse_value:.6f} rel_fro={rel_fro_value:.6f} [BEST]"
                )
            else:
                logger.info(
                    f"[eval] epoch={epoch} rel_l2={rel_l2_value:.6f} rmse={rmse_value:.6f} rel_fro={rel_fro_value:.6f}"
                )

        if should_save_running_interval:
            save_train_state(
                state,
                args.running_ckpt_dir,
                prefix=args.running_ckpt_prefix,
                step=epoch,
                max_to_keep=args.running_ckpt_max_to_keep,
            )

        loss_value = float(loss)
        loss_log.append(loss_value)
        pbar.set_postfix({"loss": loss_value})
    print("Training complete.")

    return {
        "state": state,
        "best_state": best_state,
        "best_l2_error": best_l2_error,
        "best_epoch": best_epoch,
        "loss_log": loss_log,
    }


__all__ = ["train_cfo", "train_ar", "CFOTrainArgs", "ARTrainArgs"]
