import argparse
from pathlib import Path

import h5py
import numpy as np
from clawpack import pyclaw, riemann


def create_solver(nx, ny, gravity, dam_radius):
    """
    Create one 2D radial dam-break shallow-water simulation.
    """

    # ---------------------------------------------------------
    # 1. Shallow-water Riemann solver
    # ---------------------------------------------------------
    rp = riemann.shallow_roe_with_efix_2D

    solver = pyclaw.ClawSolver2D(rp)

    solver.limiters = pyclaw.limiters.tvd.MC

    solver.num_waves = 3
    solver.num_eqn = 3

    # Extrapolation boundary conditions
    solver.bc_lower[0] = pyclaw.BC.extrap
    solver.bc_upper[0] = pyclaw.BC.extrap

    solver.bc_lower[1] = pyclaw.BC.extrap
    solver.bc_upper[1] = pyclaw.BC.extrap

    # ---------------------------------------------------------
    # 2. Spatial domain
    # ---------------------------------------------------------
    xlower = -2.5
    xupper = 2.5

    ylower = -2.5
    yupper = 2.5

    x = pyclaw.Dimension(
        xlower,
        xupper,
        nx,
        name="x",
    )

    y = pyclaw.Dimension(
        ylower,
        yupper,
        ny,
        name="y",
    )

    domain = pyclaw.Domain([x, y])

    # ---------------------------------------------------------
    # 3. State:
    #
    # q[0] = h
    # q[1] = hu
    # q[2] = hv
    # ---------------------------------------------------------
    state = pyclaw.State(
        domain,
        solver.num_eqn,
    )

    state.problem_data["grav"] = gravity

    # ---------------------------------------------------------
    # 4. Initial radial dam
    # ---------------------------------------------------------
    X, Y = state.p_centers

    radius = np.sqrt(X**2 + Y**2)

    h_inside = 2.0
    h_outside = 1.0

    state.q[0, :, :] = np.where(
        radius <= dam_radius,
        h_inside,
        h_outside,
    )

    # Initially stationary
    state.q[1, :, :] = 0.0
    state.q[2, :, :] = 0.0

    solution = pyclaw.Solution(
        state,
        domain,
    )

    return solver, solution, domain


def run_trajectory(
    nx,
    ny,
    n_time_steps,
    final_time,
    gravity,
    dam_radius,
):
    """
    Generate one trajectory.

    Output shape:
        (Nt + 1, Nx, Ny, 3)

    channels:
        0 -> h
        1 -> hu
        2 -> hv
    """

    solver, solution, domain = create_solver(
        nx=nx,
        ny=ny,
        gravity=gravity,
        dam_radius=dam_radius,
    )

    times = np.linspace(
        0.0,
        final_time,
        n_time_steps + 1,
    )

    data = np.zeros(
        (
            n_time_steps + 1,
            nx,
            ny,
            3,
        ),
        dtype=np.float32,
    )

    # Initial state
    data[0] = np.moveaxis(
        solution.state.q,
        0,
        -1,
    )

    # ---------------------------------------------------------
    # Integrate forward in physical time
    # ---------------------------------------------------------
    for k in range(
        1,
        n_time_steps + 1,
    ):

        solver.evolve_to_time(
            solution,
            times[k],
        )

        data[k] = np.moveaxis(
            solution.state.q,
            0,
            -1,
        )

    x = np.asarray(
        domain.grid.x.centers,
        dtype=np.float32,
    )

    y = np.asarray(
        domain.grid.y.centers,
        dtype=np.float32,
    )

    return data, x, y, times.astype(np.float32)


def generate_dataset(
    output_path,
    n_trajectories,
    nx,
    ny,
    n_time_steps,
    final_time,
    gravity,
    seed_start,
):
    """
    Generate multiple radial dam-break trajectories.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        raise FileExistsError(
            f"\nOutput file already exists:\n{output_path}\n"
            "Delete it manually if you want to regenerate it."
        )

    rng = np.random.default_rng(
        seed_start
    )

    with h5py.File(
        output_path,
        "w",
    ) as f:

        for i in range(
            n_trajectories
        ):

            # Same radius range used by PDEBench
            dam_radius = rng.uniform(
                0.3,
                0.7,
            )

            print(
                f"\nTrajectory "
                f"{i + 1}/{n_trajectories}"
            )

            print(
                f"Dam radius = "
                f"{dam_radius:.4f}"
            )

            data, x, y, times = run_trajectory(
                nx=nx,
                ny=ny,
                n_time_steps=n_time_steps,
                final_time=final_time,
                gravity=gravity,
                dam_radius=dam_radius,
            )

            seed_name = str(
                seed_start + i
            ).zfill(4)

            group = f.create_group(
                seed_name
            )

            group.create_dataset(
                "data",
                data=data,
                dtype="f",
            )

            grid = group.create_group(
                "grid"
            )

            grid.create_dataset(
                "x",
                data=x,
                dtype="f",
            )

            grid.create_dataset(
                "y",
                data=y,
                dtype="f",
            )

            grid.create_dataset(
                "t",
                data=times,
                dtype="f",
            )

            # Useful metadata
            group.attrs[
                "dam_radius"
            ] = dam_radius

            group.attrs[
                "gravity"
            ] = gravity

            group.attrs[
                "nx"
            ] = nx

            group.attrs[
                "ny"
            ] = ny

            group.attrs[
                "channels"
            ] = "h,hu,hv"

    print("\n================================")
    print("Dataset generation complete")
    print("================================")

    print(
        f"Saved to:\n{output_path}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--ntraj",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--nx",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--ny",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--nt",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--T",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--gravity",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    generate_dataset(
        output_path=args.output,
        n_trajectories=args.ntraj,
        nx=args.nx,
        ny=args.ny,
        n_time_steps=args.nt,
        final_time=args.T,
        gravity=args.gravity,
        seed_start=args.seed_start,
    )


if __name__ == "__main__":
    main()