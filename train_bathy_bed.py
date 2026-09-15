"""
Specialized trainer for BathymetryBedRegularizedPICFO.

This file exists so the original train.py remains untouched.

Tracks:

    total
    CFO
    PDE
    bed
    weighted PDE
    weighted bed

and keeps the same validation/checkpoint behavior as the existing
CFO trainer.
"""

from __future__ import annotations

import jax

from jax import value_and_grad
from tqdm.auto import trange

from train import (
    CFOTrainArgs,
    _prepare_cfo_train_batch,
    _run_cfo_eval,
    init_cfo_train_state,
)

from utils.checkpoints import (
    save_train_state,
)

from utils.data import (
    prepare_tf_data,
    prefetch_to_device,
)

from utils.logging import (
    ExperimentLogger,
)


def train_bathy_bed_picfo(
    method,
    spline_dataloader,
    args: CFOTrainArgs,
    eval_dataset=None,
):
    """
    Train

        L
        =
        L_CFO
        +
        lambda_PDE L_PDE
        +
        lambda_bed L_bed
    """

    # ============================================================
    # INITIALIZE MODEL
    # ============================================================

    state = init_cfo_train_state(
        method,
        seed=args.random_seed,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
    )

    # ============================================================
    # LOSS FUNCTION
    # ============================================================

    def component_loss_fn(
        params,
        batch,
    ):

        (
            total_loss,
            cfo_loss,
            pde_loss,
            bed_loss,
        ) = method.loss_components(
            params,
            batch,
        )

        return (
            total_loss,
            (
                cfo_loss,
                pde_loss,
                bed_loss,
            ),
        )

    # ============================================================
    # TRAIN STEP
    # ============================================================

    def train_step(
        state,
        batch,
    ):

        (
            (
                total_loss,
                (
                    cfo_loss,
                    pde_loss,
                    bed_loss,
                ),
            ),
            grads,
        ) = value_and_grad(
            component_loss_fn,
            has_aux=True,
        )(
            state.params,
            batch,
        )

        state = state.apply_gradients(
            grads=grads
        )

        return (
            total_loss,
            cfo_loss,
            pde_loss,
            bed_loss,
            state,
        )

    step_jit = jax.jit(
        train_step
    )

    # ============================================================
    # LOGGER
    # ============================================================

    logger = ExperimentLogger.create(
        use_wandb=args.use_wandb,
        log_mode=args.log_mode,
        train_log_interval=(
            args.train_log_interval
        ),
        console=(
            args.console_logging
        ),
    )

    logger.info(
        "Training mode: "
        "Bathymetry PI-CFO + bed-response loss"
    )

    logger.info(
        "Objective: "
        "CFO + lambda_PDE*PDE + lambda_bed*BED"
    )

    logger.info(
        f"lambda_PDE: "
        f"{float(method.lambda_pde)}"
    )

    logger.info(
        f"lambda_bed: "
        f"{float(method.lambda_bed)}"
    )

    # ============================================================
    # DATA
    # ============================================================

    data = map(
        prepare_tf_data,
        spline_dataloader,
    )

    data = prefetch_to_device(
        data,
        2,
    )

    # ============================================================
    # RANDOM KEY
    # ============================================================

    rng_key = jax.random.PRNGKey(
        args.random_seed
    )

    # ============================================================
    # LOGS
    # ============================================================

    total_loss_log = []

    cfo_loss_log = []

    pde_loss_log = []

    bed_loss_log = []

    weighted_pde_loss_log = []

    weighted_bed_loss_log = []

    # ============================================================
    # BEST VALIDATION MODEL
    # ============================================================

    best_state = state

    best_l2_error = float(
        "inf"
    )

    best_epoch = -1

    # ============================================================
    # LOOP
    # ============================================================

    pbar = trange(
        args.num_epochs,
        desc="Training",
    )

    for epoch in pbar:

        (
            rng_key,
            time_key,
            noise_key,
        ) = jax.random.split(
            rng_key,
            3,
        )

        batch = _prepare_cfo_train_batch(
            method,
            next(
                data
            ),
            time_key,
            noise_key,
        )

        (
            total_loss,
            cfo_loss,
            pde_loss,
            bed_loss,
            state,
        ) = step_jit(
            state,
            batch,
        )

        total_value = float(
            total_loss
        )

        cfo_value = float(
            cfo_loss
        )

        pde_value = float(
            pde_loss
        )

        bed_value = float(
            bed_loss
        )

        weighted_pde = (
            float(
                method.lambda_pde
            )
            *
            pde_value
        )

        weighted_bed = (
            float(
                method.lambda_bed
            )
            *
            bed_value
        )

        total_loss_log.append(
            total_value
        )

        cfo_loss_log.append(
            cfo_value
        )

        pde_loss_log.append(
            pde_value
        )

        bed_loss_log.append(
            bed_value
        )

        weighted_pde_loss_log.append(
            weighted_pde
        )

        weighted_bed_loss_log.append(
            weighted_bed
        )

        pbar.set_postfix(
            {
                "total":
                    f"{total_value:.3e}",

                "cfo":
                    f"{cfo_value:.3e}",

                "pde":
                    f"{pde_value:.3e}",

                "bed":
                    f"{bed_value:.3e}",

                "wpde":
                    f"{weighted_pde:.2e}",

                "wbed":
                    f"{weighted_bed:.2e}",
            }
        )

        logger.log(
            {
                "train/total_loss":
                    total_value,

                "train/cfo_loss":
                    cfo_value,

                "train/pde_loss":
                    pde_value,

                "train/bed_loss":
                    bed_value,

                "train/weighted_pde":
                    weighted_pde,

                "train/weighted_bed":
                    weighted_bed,

                "train/epoch":
                    epoch,
            },
            commit=False,
        )

        # ========================================================
        # RUNNING CHECKPOINT?
        # ========================================================

        should_save_running = (
            args.running_ckpt_dir
            is not None
            and
            args.running_ckpt_interval
            > 0
            and
            (
                (
                    epoch
                    +
                    1
                )
                %
                args.running_ckpt_interval
                ==
                0
            )
        )

        # ========================================================
        # VALIDATION
        # ========================================================

        if (
            args.do_eval
            and
            epoch > 0
            and
            (
                epoch
                %
                args.eval_interval
                ==
                0
            )
            and
            eval_dataset
            is not None
        ):

            (
                rel_l2_value,
                rmse_value,
                rel_fro_value,
            ) = _run_cfo_eval(
                method,
                state,
                epoch,
                eval_dataset,
                logger,
            )

            # Make sure there is a running checkpoint at eval epochs.
            should_save_running = (
                should_save_running
                or
                (
                    args.running_ckpt_dir
                    is not None
                )
            )

            if (
                rel_l2_value
                <
                best_l2_error
            ):

                best_l2_error = (
                    rel_l2_value
                )

                best_state = (
                    state
                )

                best_epoch = (
                    epoch
                )

                if (
                    args.best_ckpt_dir
                    is not None
                ):

                    save_train_state(
                        best_state,
                        args.best_ckpt_dir,
                        prefix=(
                            args.best_ckpt_prefix
                        ),
                        step=epoch,
                        max_to_keep=1,
                    )

                logger.info(
                    f"[eval] "
                    f"epoch={epoch} "
                    f"rel_l2="
                    f"{rel_l2_value:.6f} "
                    f"rmse="
                    f"{rmse_value:.6f} "
                    f"rel_fro="
                    f"{rel_fro_value:.6f} "
                    f"[BEST]"
                )

            else:

                logger.info(
                    f"[eval] "
                    f"epoch={epoch} "
                    f"rel_l2="
                    f"{rel_l2_value:.6f} "
                    f"rmse="
                    f"{rmse_value:.6f} "
                    f"rel_fro="
                    f"{rel_fro_value:.6f}"
                )

        # ========================================================
        # SAVE RUNNING
        # ========================================================

        if should_save_running:

            save_train_state(
                state,
                args.running_ckpt_dir,
                prefix=(
                    args.running_ckpt_prefix
                ),
                step=epoch,
                max_to_keep=(
                    args.running_ckpt_max_to_keep
                ),
            )

    # ============================================================
    # END
    # ============================================================

    print(
        "\n========================================"
    )

    print(
        "FINAL BED-REGULARIZED PI-CFO LOSSES"
    )

    print(
        "========================================"
    )

    print(
        "Total loss:        "
        f"{total_loss_log[-1]:.6e}"
    )

    print(
        "CFO loss:          "
        f"{cfo_loss_log[-1]:.6e}"
    )

    print(
        "PDE loss:          "
        f"{pde_loss_log[-1]:.6e}"
    )

    print(
        "Bed loss:          "
        f"{bed_loss_log[-1]:.6e}"
    )

    print(
        "Weighted PDE:      "
        f"{weighted_pde_loss_log[-1]:.6e}"
    )

    print(
        "Weighted bed:      "
        f"{weighted_bed_loss_log[-1]:.6e}"
    )

    print(
        "lambda_PDE:        "
        f"{float(method.lambda_pde):.6e}"
    )

    print(
        "lambda_bed:        "
        f"{float(method.lambda_bed):.6e}"
    )

    reconstructed = (
        cfo_loss_log[
            -1
        ]
        +
        weighted_pde_loss_log[
            -1
        ]
        +
        weighted_bed_loss_log[
            -1
        ]
    )

    print(
        "CFO + wPDE + wBED: "
        f"{reconstructed:.6e}"
    )

    print(
        "========================================"
    )

    return {
        "state":
            state,

        "best_state":
            best_state,

        "best_l2_error":
            best_l2_error,

        "best_epoch":
            best_epoch,

        "loss_log":
            total_loss_log,

        "total_loss_log":
            total_loss_log,

        "cfo_loss_log":
            cfo_loss_log,

        "pde_loss_log":
            pde_loss_log,

        "bed_loss_log":
            bed_loss_log,

        "weighted_pde_loss_log":
            weighted_pde_loss_log,

        "weighted_bed_loss_log":
            weighted_bed_loss_log,
    }


__all__ = [
    "train_bathy_bed_picfo",
]