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

from utils.data import (
    prepare_tf_data,
    prefetch_to_device,
)

from utils.checkpoints import (
    save_train_state,
)

from utils.logging import (
    ExperimentLogger,
)

from utils.metrics import (
    relative_L2_error,
    relative_frobenius_error,
    rmse,
)


# ================================================================
# CFO TRAINING ARGUMENTS
# ================================================================

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


# ================================================================
# AUTOREGRESSIVE TRAINING ARGUMENTS
# ================================================================

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


# ================================================================
# PREPARE CFO TRAINING BATCH
# ================================================================

def _prepare_cfo_train_batch(
    method,
    raw_batch,
    time_key,
    noise_key,
):

    # ------------------------------------------------------------
    # Conditioned CFO
    # ------------------------------------------------------------

    if method.use_condition:

        (
            spline_coef,
            t_start,
            t_end,
            condition,
            *_,
        ) = raw_batch

    # ------------------------------------------------------------
    # Unconditioned CFO
    # ------------------------------------------------------------

    else:

        (
            spline_coef,
            t_start,
            t_end,
            *_,
        ) = raw_batch

        condition = None

    # ------------------------------------------------------------
    # Remove TensorFlow dataloader wrapper dimension
    # ------------------------------------------------------------

    spline_coef = spline_coef[0]

    t_start = t_start[0]

    t_end = t_end[0]

    # ------------------------------------------------------------
    # Duration of each temporal spline interval
    # ------------------------------------------------------------

    dt = (
        t_end
        -
        t_start
    )

    # ------------------------------------------------------------
    # Random physical time inside each spline interval
    # ------------------------------------------------------------

    delta_t = jax.random.uniform(
        time_key,
        shape=(
            len(dt),
        ),
        minval=0.0,
        maxval=dt,
    )

    # ------------------------------------------------------------
    # CFO stochastic perturbation
    # ------------------------------------------------------------

    x0 = spline_coef[
        :,
        0,
    ]

    eps = jax.random.normal(
        noise_key,
        x0.shape,
    )

    # ------------------------------------------------------------
    # Conditioned batch
    # ------------------------------------------------------------

    if method.use_condition:

        condition = condition[
            0
        ]

        return (
            spline_coef,
            condition,
            t_start,
            t_end,
            delta_t,
            eps,
        )

    # ------------------------------------------------------------
    # Unconditioned batch
    # ------------------------------------------------------------

    return (
        spline_coef,
        t_start,
        t_end,
        delta_t,
        eps,
    )


# ================================================================
# CFO EVALUATION
# ================================================================

def _run_cfo_eval(
    method,
    state,
    epoch: int,
    eval_dataset,
    logger: ExperimentLogger,
):

    # ------------------------------------------------------------
    # Conditioned evaluation
    # ------------------------------------------------------------

    if method.use_condition:

        (
            x0_eval,
            target_eval,
            condition_eval,
        ) = eval_dataset

    # ------------------------------------------------------------
    # Standard evaluation
    # ------------------------------------------------------------

    else:

        (
            x0_eval,
            target_eval,
        ) = eval_dataset

        condition_eval = None

    # ------------------------------------------------------------
    # Continuous-time rollout
    # ------------------------------------------------------------

    pred_eval = method.uniform_inference(
        state,
        x0_eval,
        trajectory_points_num=(
            target_eval.shape[1]
        ),
        steps_per_segment=2,
        condition=condition_eval,
        method="RK4",
    )

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    rel_fro_value = (
        relative_frobenius_error(
            target_eval,
            pred_eval,
        )
    )

    rmse_value = rmse(
        target_eval,
        pred_eval,
    )

    rel_l2_value = (
        relative_L2_error(
            target_eval,
            pred_eval,
        )
    )

    # ------------------------------------------------------------
    # Evaluation logger
    # ------------------------------------------------------------

    logger.log(
        {
            "eval/epoch":
                epoch,

            "eval/Relative_L2_Error":
                rel_l2_value,

            "eval/RMSE":
                rmse_value,

            "eval/Relative_Frobenius_Error":
                rel_fro_value,
        },
        commit=False,
    )

    return (
        float(
            rel_l2_value
        ),
        float(
            rmse_value
        ),
        float(
            rel_fro_value
        ),
    )


# ================================================================
# INITIALIZE CFO TRAIN STATE
# ================================================================

def init_cfo_train_state(
    method,
    *,
    seed: int,
    learning_rate: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.99,
) -> TrainState:

    # ------------------------------------------------------------
    # Random seed
    # ------------------------------------------------------------

    rng_key = jax.random.PRNGKey(
        seed
    )

    # ------------------------------------------------------------
    # Dummy input state
    # ------------------------------------------------------------

    x = jnp.ones(
        (
            1,
        )
        +
        tuple(
            method.input_shape
        ),
        dtype=jnp.float32,
    )

    # ------------------------------------------------------------
    # Dummy time
    # ------------------------------------------------------------

    t = jnp.ones(
        (
            1,
        ),
        dtype=jnp.float32,
    )

    # ------------------------------------------------------------
    # Conditioned model initialization
    # ------------------------------------------------------------

    if method.use_condition:

        c = jnp.ones(
            (
                1,
            )
            +
            tuple(
                method.condition_shape
            ),
            dtype=jnp.float32,
        )

        variables = method.model.init(
            rng_key,
            x,
            t,
            c,
        )

    # ------------------------------------------------------------
    # Standard model initialization
    # ------------------------------------------------------------

    else:

        variables = method.model.init(
            rng_key,
            x,
            t,
        )

    # ------------------------------------------------------------
    # Adam optimizer
    # ------------------------------------------------------------

    tx = optax.adam(
        learning_rate=float(
            learning_rate
        ),
        b1=float(
            beta1
        ),
        b2=float(
            beta2
        ),
    )

    # ------------------------------------------------------------
    # Flax TrainState
    # ------------------------------------------------------------

    return TrainState.create(
        apply_fn=method.model.apply,
        params=variables[
            "params"
        ],
        tx=tx,
    )


# ================================================================
# CFO / PI-CFO TRAINING
# ================================================================

def train_cfo(
    method,
    spline_dataloader,
    args: CFOTrainArgs,
    eval_dataset: Optional[
        Tuple[
            np.ndarray,
            np.ndarray,
        ]
    ] = None,
):
    """Train either standard CFO or Physics-Informed CFO.

    Standard CFO
    ------------
    Optimizes:

        L_CFO

    Physics-Informed CFO
    --------------------
    If the supplied method has ``loss_components()``,
    this function automatically optimizes:

        L_total
        =
        L_CFO
        +
        lambda_PDE * L_PDE

    while separately tracking:

        L_total
        L_CFO
        L_PDE
        lambda_PDE * L_PDE
    """

    # ============================================================
    # INITIALIZE TRAIN STATE
    # ============================================================

    state = init_cfo_train_state(
        method,
        seed=args.random_seed,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
    )

    # ============================================================
    # DETECT PHYSICS-INFORMED CFO
    # ============================================================

    use_loss_components = hasattr(
        method,
        "loss_components",
    )

    # ============================================================
    # PI-CFO TRAIN STEP
    # ============================================================

    if use_loss_components:

        # --------------------------------------------------------
        # loss_components() returns:
        #
        # total_loss
        # cfo_loss
        # pde_loss
        #
        # Only total_loss is differentiated.
        # --------------------------------------------------------

        def component_loss_fn(
            params,
            batch,
        ):

            (
                total_loss,
                cfo_loss,
                pde_loss,
            ) = method.loss_components(
                params,
                batch,
            )

            return (
                total_loss,
                (
                    cfo_loss,
                    pde_loss,
                ),
            )

        def cfo_train_step(
            state: TrainState,
            batch,
        ):

            (
                (
                    total_loss,
                    (
                        cfo_loss,
                        pde_loss,
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

            state = (
                state.apply_gradients(
                    grads=grads
                )
            )

            return (
                total_loss,
                cfo_loss,
                pde_loss,
                state,
            )

    # ============================================================
    # STANDARD CFO TRAIN STEP
    # ============================================================

    else:

        loss_fn = (
            method.loss_fn
        )

        def cfo_train_step(
            state: TrainState,
            batch,
        ):

            (
                loss,
                grads,
            ) = value_and_grad(
                loss_fn
            )(
                state.params,
                batch,
            )

            state = (
                state.apply_gradients(
                    grads=grads
                )
            )

            return (
                loss,
                state,
            )

    # ============================================================
    # JIT COMPILE TRAIN STEP
    # ============================================================

    step_jit = jax.jit(
        cfo_train_step
    )

    # ============================================================
    # EXPERIMENT LOGGER
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

    # ============================================================
    # MODEL PARAMETER COUNT
    # ============================================================

    num_params = sum(
        x.size
        for x in tree_leaves(
            state.params
        )
    )

    logger.info(
        f"Model parameters: "
        f"{int(num_params)}"
    )

    # ------------------------------------------------------------
    # Print training mode
    # ------------------------------------------------------------

    if use_loss_components:

        logger.info(
            "Training mode: "
            "Physics-Informed CFO"
        )

        logger.info(
            "Tracking losses: "
            "total, CFO, PDE, weighted PDE"
        )

        logger.info(
            "lambda_PDE: "
            f"{float(method.lambda_pde)}"
        )

    else:

        logger.info(
            "Training mode: "
            "Standard CFO"
        )

    # ============================================================
    # RANDOM NUMBER GENERATOR
    # ============================================================

    rng_key = (
        jax.random.PRNGKey(
            args.random_seed
        )
    )

    # ============================================================
    # PROGRESS BAR
    # ============================================================

    pbar = trange(
        args.num_epochs,
        desc="Training",
    )

    # ============================================================
    # DATA PIPELINE
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
    # LOSS LOGS
    # ============================================================

    total_loss_log: list[
        float
    ] = []

    cfo_loss_log: list[
        float
    ] = []

    pde_loss_log: list[
        float
    ] = []

    weighted_pde_loss_log: list[
        float
    ] = []

    # ============================================================
    # BEST CHECKPOINT TRACKING
    # ============================================================

    best_state = state

    best_l2_error = float(
        "inf"
    )

    best_epoch = -1

    # ============================================================
    # TRAINING LOOP
    # ============================================================

    for epoch in pbar:

        # --------------------------------------------------------
        # Random keys
        # --------------------------------------------------------

        (
            rng_key,
            time_key,
            noise_key,
        ) = jax.random.split(
            rng_key,
            3,
        )

        # --------------------------------------------------------
        # Construct random-time CFO training batch
        # --------------------------------------------------------

        batch = _prepare_cfo_train_batch(
            method,
            next(
                data
            ),
            time_key,
            noise_key,
        )

        # ========================================================
        # PHYSICS-INFORMED CFO
        # ========================================================

        if use_loss_components:

            (
                total_loss,
                cfo_loss,
                pde_loss,
                state,
            ) = step_jit(
                state,
                batch,
            )

            # ----------------------------------------------------
            # Convert JAX scalars to Python floats
            # ----------------------------------------------------

            total_loss_value = float(
                total_loss
            )

            cfo_loss_value = float(
                cfo_loss
            )

            pde_loss_value = float(
                pde_loss
            )

            weighted_pde_value = (
                float(
                    method.lambda_pde
                )
                *
                pde_loss_value
            )

            # ----------------------------------------------------
            # Save loss history
            # ----------------------------------------------------

            total_loss_log.append(
                total_loss_value
            )

            cfo_loss_log.append(
                cfo_loss_value
            )

            pde_loss_log.append(
                pde_loss_value
            )

            weighted_pde_loss_log.append(
                weighted_pde_value
            )

            # ----------------------------------------------------
            # Progress bar
            # ----------------------------------------------------

            pbar.set_postfix(
                {
                    "total":
                        f"{total_loss_value:.3e}",

                    "cfo":
                        f"{cfo_loss_value:.3e}",

                    "pde":
                        f"{pde_loss_value:.3e}",

                    "wpde":
                        f"{weighted_pde_value:.3e}",
                }
            )

        # ========================================================
        # STANDARD CFO
        # ========================================================

        else:

            (
                loss,
                state,
            ) = step_jit(
                state,
                batch,
            )

            loss_value = float(
                loss
            )

            # ----------------------------------------------------
            # Standard CFO:
            #
            # total loss = CFO loss
            # ----------------------------------------------------

            total_loss_log.append(
                loss_value
            )

            cfo_loss_log.append(
                loss_value
            )

            pbar.set_postfix(
                {
                    "loss":
                        f"{loss_value:.3e}"
                }
            )

        # ========================================================
        # RUNNING CHECKPOINT CONDITION
        # ========================================================

        should_save_running_interval = (
            args.running_ckpt_dir
            is not None
            and
            args.running_ckpt_interval
            > 0
            and
            (
                (
                    epoch + 1
                )
                %
                args.running_ckpt_interval
                ==
                0
            )
        )

        # ========================================================
        # PERIODIC EVALUATION
        # ========================================================

        if (
            args.do_eval
            and
            (
                epoch
                %
                args.eval_interval
                ==
                0
            )
            and
            epoch > 0
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

            should_save_running_interval = (
                should_save_running_interval
                or
                (
                    args.running_ckpt_dir
                    is not None
                )
            )

            # ----------------------------------------------------
            # New best validation checkpoint
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # Evaluation without new best
            # ----------------------------------------------------

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
        # SAVE RUNNING CHECKPOINT
        # ========================================================

        if should_save_running_interval:

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
    # TRAINING COMPLETE
    # ============================================================

    print(
        "Training complete."
    )

    # ============================================================
    # FINAL PI-CFO LOSS SUMMARY
    # ============================================================

    if (
        use_loss_components
        and
        len(
            total_loss_log
        )
        > 0
    ):

        print(
            "\n========================================"
        )

        print(
            "FINAL PI-CFO TRAINING LOSSES"
        )

        print(
            "========================================"
        )

        print(
            "Total loss: "
            f"{total_loss_log[-1]:.6e}"
        )

        print(
            "CFO loss:   "
            f"{cfo_loss_log[-1]:.6e}"
        )

        print(
            "PDE loss:   "
            f"{pde_loss_log[-1]:.6e}"
        )

        print(
            "Weighted PDE loss: "
            f"{weighted_pde_loss_log[-1]:.6e}"
        )

        print(
            "lambda_PDE: "
            f"{float(method.lambda_pde):.6e}"
        )

        # --------------------------------------------------------
        # Verify decomposition numerically
        # --------------------------------------------------------

        reconstructed_total = (
            cfo_loss_log[-1]
            +
            weighted_pde_loss_log[-1]
        )

        print(
            "CFO + weighted PDE: "
            f"{reconstructed_total:.6e}"
        )

        print(
            "========================================"
        )

    # ============================================================
    # RETURN TRAINING RESULTS
    # ============================================================

    return {

        "state":
            state,

        "best_state":
            best_state,

        "best_l2_error":
            best_l2_error,

        "best_epoch":
            best_epoch,

        # --------------------------------------------------------
        # Preserve old CFO API
        # --------------------------------------------------------

        "loss_log":
            total_loss_log,

        # --------------------------------------------------------
        # New Phase-29 component logs
        # --------------------------------------------------------

        "total_loss_log":
            total_loss_log,

        "cfo_loss_log":
            cfo_loss_log,

        "pde_loss_log":
            pde_loss_log,

        "weighted_pde_loss_log":
            weighted_pde_loss_log,
    }


# ================================================================
# PREPARE AUTOREGRESSIVE TRAINING BATCH
# ================================================================

def _prepare_ar_train_batch(
    method,
    raw_batch,
):

    if method.use_time:

        (
            x,
            y,
            t,
            *_,
        ) = raw_batch

        return (
            x[0],
            y[0],
            t[0],
        )

    (
        x,
        y,
        *_,
    ) = raw_batch

    return (
        x[0],
        y[0],
    )


# ================================================================
# AUTOREGRESSIVE EVALUATION
# ================================================================

def _run_ar_eval(
    method,
    state,
    epoch: int,
    eval_dataset,
    logger: ExperimentLogger,
):

    (
        x0_eval,
        target_eval,
    ) = eval_dataset

    # ------------------------------------------------------------
    # AR inference
    # ------------------------------------------------------------

    pred_eval = method.inference(
        state,
        x0_eval,
    )

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    rel_fro_value = (
        relative_frobenius_error(
            target_eval,
            pred_eval,
        )
    )

    rmse_value = rmse(
        target_eval,
        pred_eval,
    )

    rel_l2_value = (
        relative_L2_error(
            target_eval,
            pred_eval,
        )
    )

    # ------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------

    logger.log(
        {
            "eval/epoch":
                epoch,

            "eval/Relative_L2_Error":
                rel_l2_value,

            "eval/RMSE":
                rmse_value,

            "eval/Relative_Frobenius_Error":
                rel_fro_value,
        },
        commit=False,
    )

    return (
        float(
            rel_l2_value
        ),
        float(
            rmse_value
        ),
        float(
            rel_fro_value
        ),
    )


# ================================================================
# INITIALIZE AR TRAIN STATE
# ================================================================

def init_ar_train_state(
    method,
    *,
    seed: int,
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.99,
) -> TrainState:

    # ------------------------------------------------------------
    # RNG
    # ------------------------------------------------------------

    rng_key = (
        jax.random.PRNGKey(
            seed
        )
    )

    # ------------------------------------------------------------
    # Dummy state
    # ------------------------------------------------------------

    x = jnp.ones(
        (
            1,
        )
        +
        tuple(
            method.input_shape
        ),
        dtype=jnp.float32,
    )

    # ------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------

    if method.use_time:

        t = jnp.ones(
            (
                1,
            ),
            dtype=jnp.float32,
        )

        variables = (
            method.model.init(
                rng_key,
                x,
                t,
            )
        )

    else:

        variables = (
            method.model.init(
                rng_key,
                x,
            )
        )

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------

    tx = optax.adam(
        learning_rate=float(
            learning_rate
        ),
        b1=float(
            beta1
        ),
        b2=float(
            beta2
        ),
    )

    # ------------------------------------------------------------
    # Train state
    # ------------------------------------------------------------

    return TrainState.create(
        apply_fn=(
            method.model.apply
        ),
        params=(
            variables[
                "params"
            ]
        ),
        tx=tx,
    )


# ================================================================
# AUTOREGRESSIVE TRAINING
# ================================================================

def train_ar(
    method,
    dataloader,
    args: ARTrainArgs,
    eval_dataset: Optional[
        Tuple[
            np.ndarray,
            np.ndarray,
        ]
    ] = None,
):

    # ============================================================
    # INITIALIZE STATE
    # ============================================================

    state = init_ar_train_state(
        method,
        seed=args.random_seed,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
    )

    # ============================================================
    # LOSS
    # ============================================================

    loss_fn = (
        method.loss_fn
    )

    def ar_train_step(
        state: TrainState,
        batch,
    ):

        (
            loss,
            grads,
        ) = value_and_grad(
            loss_fn
        )(
            state.params,
            batch,
        )

        state = (
            state.apply_gradients(
                grads=grads
            )
        )

        return (
            loss,
            state,
        )

    # ============================================================
    # JIT
    # ============================================================

    step_jit = jax.jit(
        ar_train_step
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

    # ============================================================
    # PARAMETER COUNT
    # ============================================================

    num_params = sum(
        x.size
        for x in tree_leaves(
            state.params
        )
    )

    logger.info(
        f"Model parameters: "
        f"{int(num_params)}"
    )

    # ============================================================
    # PROGRESS BAR
    # ============================================================

    pbar = trange(
        args.num_epochs,
        desc="Training",
    )

    # ============================================================
    # DATA
    # ============================================================

    data = map(
        prepare_tf_data,
        dataloader,
    )

    data = prefetch_to_device(
        data,
        2,
    )

    # ============================================================
    # LOGS
    # ============================================================

    loss_log: list[
        float
    ] = []

    # ============================================================
    # BEST MODEL
    # ============================================================

    best_state = state

    best_l2_error = float(
        "inf"
    )

    best_epoch = -1

    # ============================================================
    # TRAIN LOOP
    # ============================================================

    for epoch in pbar:

        # --------------------------------------------------------
        # Batch
        # --------------------------------------------------------

        batch = (
            _prepare_ar_train_batch(
                method,
                next(
                    data
                ),
            )
        )

        # --------------------------------------------------------
        # Optimization
        # --------------------------------------------------------

        (
            loss,
            state,
        ) = step_jit(
            state,
            batch,
        )

        # --------------------------------------------------------
        # Running checkpoint condition
        # --------------------------------------------------------

        should_save_running_interval = (
            args.running_ckpt_dir
            is not None
            and
            args.running_ckpt_interval
            > 0
            and
            (
                (
                    epoch + 1
                )
                %
                args.running_ckpt_interval
                ==
                0
            )
        )

        # --------------------------------------------------------
        # Evaluation
        # --------------------------------------------------------

        if (
            args.do_eval
            and
            (
                epoch
                %
                args.eval_interval
                ==
                0
            )
            and
            epoch > 0
            and
            eval_dataset
            is not None
        ):

            (
                rel_l2_value,
                rmse_value,
                rel_fro_value,
            ) = _run_ar_eval(
                method,
                state,
                epoch,
                eval_dataset,
                logger,
            )

            should_save_running_interval = (
                should_save_running_interval
                or
                (
                    args.running_ckpt_dir
                    is not None
                )
            )

            # ----------------------------------------------------
            # New best model
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # Not best
            # ----------------------------------------------------

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

        # --------------------------------------------------------
        # Running checkpoint
        # --------------------------------------------------------

        if should_save_running_interval:

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

        # --------------------------------------------------------
        # Loss history
        # --------------------------------------------------------

        loss_value = float(
            loss
        )

        loss_log.append(
            loss_value
        )

        # --------------------------------------------------------
        # Progress bar
        # --------------------------------------------------------

        pbar.set_postfix(
            {
                "loss":
                    loss_value
            }
        )

    # ============================================================
    # FINISH
    # ============================================================

    print(
        "Training complete."
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
            loss_log,
    }


# ================================================================
# PUBLIC API
# ================================================================

__all__ = [
    "train_cfo",
    "train_ar",
    "CFOTrainArgs",
    "ARTrainArgs",
]