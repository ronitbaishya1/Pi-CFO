import jax.numpy as jnp


def swe_fluxes(q, g=1.0):

    h = q[..., 0]
    hu = q[..., 1]
    hv = q[..., 2]

    h_safe = jnp.maximum(h, 1e-6)

    fx = jnp.stack(
        [
            hu,
            hu**2 / h_safe + 0.5 * g * h**2,
            hu * hv / h_safe,
        ],
        axis=-1,
    )

    fy = jnp.stack(
        [
            hv,
            hu * hv / h_safe,
            hv**2 / h_safe + 0.5 * g * h**2,
        ],
        axis=-1,
    )

    return fx, fy


def ddx(f, dx):

    return (
        f[:, 2:, 1:-1, :]
        - f[:, :-2, 1:-1, :]
    ) / (2.0 * dx)


def ddy(f, dy):

    return (
        f[:, 1:-1, 2:, :]
        - f[:, 1:-1, :-2, :]
    ) / (2.0 * dy)


def swe_residual(
    q,
    q_t,
    dx=0.15625,
    dy=0.15625,
    g=1.0,
):

    fx, fy = swe_fluxes(q, g)

    div_x = ddx(fx, dx)
    div_y = ddy(fy, dy)

    q_t_inner = q_t[:, 1:-1, 1:-1, :]

    return q_t_inner + div_x + div_y


def swe_physics_loss(
    q,
    q_t,
    dx=0.15625,
    dy=0.15625,
    g=1.0,
):

    residual = swe_residual(
        q=q,
        q_t=q_t,
        dx=dx,
        dy=dy,
        g=g,
    )

    return jnp.mean(residual**2)