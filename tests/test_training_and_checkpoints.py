from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import local_device_count

from autoregressive import Autoregressive
from cfo import ContinuousFlowOperator
from models.mlp import SimpleMLP
from train import ARTrainArgs, CFOTrainArgs, init_ar_train_state, train_ar, train_cfo
from utils.checkpoints import load_train_state, save_train_state
from utils.data import autoregressive_dataset, build_dataloader, linear_spline


def test_train_cfo_one_epoch_smoke(tmp_path: Path):
    devices = local_device_count()
    rng = np.random.default_rng(0)
    trajectories = rng.normal(size=(2 * devices, 4, 3)).astype(np.float32)
    times = np.broadcast_to(np.linspace(0.0, 1.0, trajectories.shape[1], dtype=np.float32), trajectories.shape[:2])

    spline_coef, t_start, t_end = linear_spline(trajectories, time=times)
    loader = build_dataloader(
        X=spline_coef,
        t1=t_start,
        t2=t_end,
        batch_size=2 * devices,
        num_epochs=1,
        seed=0,
    )

    method = ContinuousFlowOperator(
        model=SimpleMLP(output_dim=3),
        input_shape=(3,),
        gamma=1e-5,
        spline_type="linear",
        use_condition=False,
    )
    args = CFOTrainArgs(num_epochs=1, random_seed=0, do_eval=False)

    out = train_cfo(method, loader, args, eval_dataset=None)
    assert len(out["loss_log"]) == 1
    assert np.isfinite(out["loss_log"][0])


def test_train_ar_one_epoch_smoke():
    devices = local_device_count()
    rng = np.random.default_rng(1)
    trajectories = rng.normal(size=(2 * devices, 5, 3)).astype(np.float32)
    x, y = autoregressive_dataset(trajectories)
    loader = build_dataloader(X=x, y=y, batch_size=2 * devices, num_epochs=1, seed=0)

    method = Autoregressive(
        model=SimpleMLP(output_dim=3),
        input_shape=(3,),
        trajectory_points_num=trajectories.shape[1],
        use_time=False,
    )
    args = ARTrainArgs(num_epochs=1, random_seed=0, do_eval=False)

    out = train_ar(method, loader, args, eval_dataset=None)
    assert len(out["loss_log"]) == 1
    assert np.isfinite(out["loss_log"][0])


def test_checkpoint_roundtrip(tmp_path: Path):
    method = Autoregressive(
        model=SimpleMLP(output_dim=3),
        input_shape=(3,),
        trajectory_points_num=5,
        use_time=False,
    )
    state = init_ar_train_state(method, seed=0, learning_rate=1e-3, beta1=0.9, beta2=0.999)

    ckpt_dir = str(tmp_path / "ckpts")
    save_train_state(state, ckpt_dir=ckpt_dir, prefix="unit", step=1, max_to_keep=1)
    restored = load_train_state(state, ckpt_dir=ckpt_dir, prefix="unit", step=1)

    leaves_a = jax.tree_util.tree_leaves(state.params)
    leaves_b = jax.tree_util.tree_leaves(restored.params)
    assert len(leaves_a) == len(leaves_b)
    for a, b in zip(leaves_a, leaves_b):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)

    assert int(restored.step) == int(state.step)
