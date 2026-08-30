"""Model factory for project-level imports."""

from __future__ import annotations

from typing import Sequence

from models.dit import DiT
from models.fno import FNO1d, FNO2d
from models.mlp import SimpleMLP
from models.unet import UNet1D, UNet2D


def build_model(
	model_name: str,
	input_shape: Sequence[int],
	use_condition: bool = False,
):
	"""Instantiate a model from a standardized model name."""
	name = model_name.strip()
	name_l = name.lower()

	out_channels = input_shape[-1] if len(input_shape) >= 2 else 1

	if name_l == "unet1d":
		return UNet1D(use_condition=use_condition)

	if name_l == "unet2d":
		return UNet2D(use_condition=use_condition)

	if name_l == "fno1d":
		return FNO1d(num_channels=out_channels, use_condition=use_condition)

	if name_l == "fno2d":
		return FNO2d(num_channels=out_channels, use_condition=use_condition)

	if name_l == "dit":
		out_channels = input_shape[-1] if len(input_shape) >= 3 else 1
		return DiT(out_channels=out_channels)

	if name_l == "simplemlp":
		output_dim = input_shape[-1] if len(input_shape) > 0 else 1
		return SimpleMLP(use_condition=use_condition, output_dim=output_dim)

	raise ValueError(
		f"Unsupported model_name='{model_name}'. "
		"Supported: UNet1D, UNet2D, FNO1d, FNO2d, DiT, SimpleMLP"
	)

__all__ = ["build_model"]
