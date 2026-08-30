import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence
import numpy as np


class Autoregressive:
	def __init__(
		self,
		*,
		model: nn.Module,
		input_shape: Sequence[int],
		trajectory_points_num: int,
		use_time: bool = False,
	):
		self.model = model
		self.input_shape = tuple(input_shape)
		self.use_time = bool(use_time)
		self.trajectory_points_num = int(trajectory_points_num)

	def _model_apply(self, params, x, t=None):
		if self.use_time:
			if t is None:
				raise ValueError("`t` must be provided when `use_time=True`.")
			return self.model.apply({'params': params}, x, t)
		return self.model.apply({'params': params}, x)

	def loss_fn(self, params, batch):
		if self.use_time:
			x, y, t = batch
		else:
			x, y = batch
			t = None
		pred = self._model_apply(params, x, t)
		return jnp.mean((pred - y) ** 2)

	def inference(self, state, x0):
		trajectory = []
		x0 = jnp.asarray(x0) if not isinstance(x0, jnp.ndarray) else x0
		if self.use_time:
			t_values = jnp.linspace(0, 1, self.trajectory_points_num)

		def one_step(x, t=None):
			return self._model_apply(state.params, x, t)

		for i in range(self.trajectory_points_num):
			trajectory.append(np.array(x0))
			if self.use_time:
				t = t_values[i]
				x0 = one_step(x0, t)
			else:
				x0 = one_step(x0)
		return np.swapaxes(np.stack(trajectory), 0, 1)
