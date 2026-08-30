import jax
import jax.numpy as jnp
import pytest

from autoregressive import Autoregressive
from cfo import ContinuousFlowOperator
from models.mlp import SimpleMLP
from train import init_ar_train_state, init_cfo_train_state


def test_cfo_loss_fn_smoke():
    model = SimpleMLP(output_dim=3)
    method = ContinuousFlowOperator(
        model=model,
        input_shape=(3,),
        gamma=1e-5,
        spline_type="linear",
        use_condition=False,
    )
    state = init_cfo_train_state(method, seed=0, learning_rate=1e-3, beta1=0.9, beta2=0.999)

    n = 8
    spline_coef = jax.random.normal(jax.random.PRNGKey(1), (n, 2, 3))
    t_start = jnp.zeros((n,), dtype=jnp.float32)
    t_end = jnp.ones((n,), dtype=jnp.float32)
    delta_t = jax.random.uniform(jax.random.PRNGKey(2), (n,), minval=0.0, maxval=1.0)
    eps = jax.random.normal(jax.random.PRNGKey(3), (n, 3))

    loss = method.loss_fn(state.params, (spline_coef, t_start, t_end, delta_t, eps))
    assert jnp.isfinite(loss)
    assert float(loss) >= 0.0


def test_ar_loss_fn_smoke():
    model = SimpleMLP(output_dim=3)
    method = Autoregressive(
        model=model,
        input_shape=(3,),
        trajectory_points_num=5,
        use_time=False,
    )
    state = init_ar_train_state(method, seed=0, learning_rate=1e-3, beta1=0.9, beta2=0.999)

    x = jax.random.normal(jax.random.PRNGKey(4), (16, 3))
    y = jax.random.normal(jax.random.PRNGKey(5), (16, 3))
    loss = method.loss_fn(state.params, (x, y))

    assert jnp.isfinite(loss)
    assert float(loss) >= 0.0


def test_cfo_infer_at_method_dispatch_and_error():
    model = SimpleMLP(output_dim=3)
    method = ContinuousFlowOperator(
        model=model,
        input_shape=(3,),
        gamma=1e-5,
        spline_type="linear",
        use_condition=False,
    )
    state = init_cfo_train_state(method, seed=0, learning_rate=1e-3, beta1=0.9, beta2=0.999)

    x0 = jnp.zeros((4, 3), dtype=jnp.float32)
    for solver in ("Euler", "Heun", "RK4"):
        out = method.infer_at(state, x0, s=0.0, t=1.0, steps=11, condition=None, method=solver)
        assert out.shape == x0.shape
        assert jnp.all(jnp.isfinite(out))

    with pytest.raises(ValueError, match="method must be"):
        method.infer_at(state, x0, s=0.0, t=1.0, steps=11, condition=None, method="BadSolver")
