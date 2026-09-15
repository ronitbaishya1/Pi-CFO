"""Phase 50 tests for bathymetry conditioning through CFO/FNO."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax.training.train_state import TrainState

from cfo import ContinuousFlowOperator

from models.fno import FNO2d

from utils.bathy_training import (
    repeat_bathymetry_for_spline_intervals,
)

from utils.data import (
    linear_spline,
)


def _small_conditioned_method():

    model = FNO2d(
        num_channels=3,
        modes1=4,
        modes2=4,
        width=8,
        num_blocks=2,
        use_condition=True,
        use_time=True,
    )

    method = ContinuousFlowOperator(
        model=model,
        input_shape=(
            8,
            8,
            3,
        ),
        gamma=1e-5,
        spline_type="linear",
        use_condition=True,
        condition_shape=(
            8,
            8,
            1,
        ),
    )

    return (
        model,
        method,
    )


def test_bathymetry_repetition_matches_trajectory_intervals():

    bathymetry = np.zeros(
        (
            3,
            4,
            4,
            1,
        ),
        dtype=np.float32,
    )

    bathymetry[
        0,
        ...,
    ] = 10.0

    bathymetry[
        1,
        ...,
    ] = 20.0

    bathymetry[
        2,
        ...,
    ] = 30.0

    repeated = (
        repeat_bathymetry_for_spline_intervals(
            bathymetry,
            num_snapshots=5,
        )
    )

    assert repeated.shape == (
        12,
        4,
        4,
        1,
    )

    np.testing.assert_allclose(
        repeated[0:4],
        10.0,
    )

    np.testing.assert_allclose(
        repeated[4:8],
        20.0,
    )

    np.testing.assert_allclose(
        repeated[8:12],
        30.0,
    )


def test_spline_flattening_and_bathymetry_order_match():

    train_data = np.zeros(
        (
            2,
            3,
            2,
            2,
            1,
        ),
        dtype=np.float32,
    )

    train_data[
        0,
        0,
        ...,
    ] = 0.0

    train_data[
        0,
        1,
        ...,
    ] = 1.0

    train_data[
        0,
        2,
        ...,
    ] = 2.0

    train_data[
        1,
        0,
        ...,
    ] = 10.0

    train_data[
        1,
        1,
        ...,
    ] = 11.0

    train_data[
        1,
        2,
        ...,
    ] = 12.0

    time = np.broadcast_to(
        np.asarray(
            [
                0.0,
                0.5,
                1.0,
            ],
            dtype=np.float32,
        )[
            None,
            :,
        ],
        (
            2,
            3,
        ),
    )

    (
        spline_coef,
        _,
        _,
    ) = linear_spline(
        train_data,
        time=time,
    )

    bathymetry = np.zeros(
        (
            2,
            2,
            2,
            1,
        ),
        dtype=np.float32,
    )

    bathymetry[
        0,
        ...,
    ] = 100.0

    bathymetry[
        1,
        ...,
    ] = 200.0

    repeated = (
        repeat_bathymetry_for_spline_intervals(
            bathymetry,
            num_snapshots=3,
        )
    )

    np.testing.assert_allclose(
        spline_coef[
            :,
            0,
            0,
            0,
            0,
        ],
        np.asarray(
            [
                0.0,
                1.0,
                10.0,
                11.0,
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_allclose(
        repeated[
            :,
            0,
            0,
            0,
        ],
        np.asarray(
            [
                100.0,
                100.0,
                200.0,
                200.0,
            ],
            dtype=np.float32,
        ),
    )


def test_fno_accepts_q_time_and_bathymetry_and_outputs_three_channels():

    (
        model,
        _,
    ) = _small_conditioned_method()

    q = jnp.ones(
        (
            2,
            8,
            8,
            3,
        ),
        dtype=jnp.float32,
    )

    t = jnp.asarray(
        [
            0.25,
            0.75,
        ],
        dtype=jnp.float32,
    )

    b = (
        jnp.ones(
            (
                2,
                8,
                8,
                1,
            ),
            dtype=jnp.float32,
        )
        *
        0.1
    )

    variables = model.init(
        jax.random.PRNGKey(
            0
        ),
        q,
        t,
        b,
    )

    output = model.apply(
        variables,
        q,
        t,
        b,
    )

    assert output.shape == q.shape

    # Input to fc0:
    #
    # 3 state channels
    # + 2 grid channels
    # + 1 bathymetry channel
    #
    # = 6 channels

    assert (
        variables[
            "params"
        ][
            "fc0"
        ][
            "kernel"
        ].shape[0]
        ==
        6
    )


def test_conditioned_cfo_loss_is_finite():

    (
        model,
        method,
    ) = _small_conditioned_method()

    batch_size = 2

    x0 = jnp.ones(
        (
            batch_size,
            8,
            8,
            3,
        ),
        dtype=jnp.float32,
    )

    x1 = (
        x0.at[
            ...,
            0,
        ]
        .add(
            0.05
        )
    )

    spline_coef = jnp.stack(
        [
            x0,
            x1 - x0,
        ],
        axis=1,
    )

    bathymetry = (
        jnp.ones(
            (
                batch_size,
                8,
                8,
                1,
            ),
            dtype=jnp.float32,
        )
        *
        0.1
    )

    t_start = jnp.zeros(
        (
            batch_size,
        ),
        dtype=jnp.float32,
    )

    t_end = jnp.ones(
        (
            batch_size,
        ),
        dtype=jnp.float32,
    )

    delta_t = jnp.asarray(
        [
            0.25,
            0.75,
        ],
        dtype=jnp.float32,
    )

    eps = jnp.zeros_like(
        x0
    )

    variables = model.init(
        jax.random.PRNGKey(
            1
        ),
        x0,
        delta_t,
        bathymetry,
    )

    batch = (
        spline_coef,
        bathymetry,
        t_start,
        t_end,
        delta_t,
        eps,
    )

    loss = method.loss_fn(
        variables[
            "params"
        ],
        batch,
    )

    assert jnp.isfinite(
        loss
    )


def test_conditioned_rk4_inference_runs_and_preserves_shape():

    (
        model,
        method,
    ) = _small_conditioned_method()

    q0 = jnp.ones(
        (
            2,
            8,
            8,
            3,
        ),
        dtype=jnp.float32,
    )

    t0 = jnp.zeros(
        (
            2,
        ),
        dtype=jnp.float32,
    )

    b = (
        jnp.ones(
            (
                2,
                8,
                8,
                1,
            ),
            dtype=jnp.float32,
        )
        *
        0.1
    )

    variables = model.init(
        jax.random.PRNGKey(
            2
        ),
        q0,
        t0,
        b,
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables[
            "params"
        ],
        tx=optax.adam(
            1e-4
        ),
    )

    pred = method.uniform_inference(
        state,
        q0,
        trajectory_points_num=3,
        steps_per_segment=1,
        condition=b,
        method="RK4",
    )

    assert pred.shape == (
        2,
        3,
        8,
        8,
        3,
    )

    assert np.isfinite(
        pred
    ).all()