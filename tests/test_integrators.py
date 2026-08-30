import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from utils.integrators import INTEGRATOR_STEP_FNS, euler_step, heun_step, rk4_step


def _make_state() -> TrainState:
    def _apply_fn(variables, x, t, condition):
        del variables, t, condition
        return jnp.ones_like(x) * 2.0

    return TrainState.create(
        apply_fn=_apply_fn,
        params={"dummy": jnp.array(0.0, dtype=jnp.float32)},
        tx=optax.sgd(learning_rate=0.0),
    )


def test_single_steps_constant_velocity():
    state = _make_state()
    x0 = jnp.zeros((2, 3), dtype=jnp.float32)
    t = jnp.zeros((2,), dtype=jnp.float32)
    dt = 0.1

    e = euler_step(state, x0, t, None, dt)
    h = heun_step(state, x0, t, None, dt)
    r = rk4_step(state, x0, t, None, dt)

    expected = np.full((2, 3), 0.2, dtype=np.float32)
    np.testing.assert_allclose(np.array(e), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(h), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(r), expected, rtol=1e-6, atol=1e-6)


def test_integrator_step_dispatch_map():
    state = _make_state()
    x0 = jnp.zeros((4, 2), dtype=jnp.float32)
    t = jnp.zeros((4,), dtype=jnp.float32)
    dt = 0.1

    assert set(INTEGRATOR_STEP_FNS.keys()) == {"Euler", "RK4", "Heun"}
    for name in ["Euler", "Heun", "RK4"]:
        step_fn = INTEGRATOR_STEP_FNS[name]
        out = step_fn(state, x0, t, None, dt)
        np.testing.assert_allclose(np.array(out), np.full((4, 2), 0.2, dtype=np.float32), rtol=1e-6, atol=1e-6)
