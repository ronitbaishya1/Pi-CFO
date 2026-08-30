from __future__ import annotations

"""Lightweight experiment logging abstraction with optional W&B support."""

from dataclasses import dataclass
from typing import Any, Mapping


def _to_float_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Convert metric values to float where possible, dropping non-numeric entries."""
    out: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


@dataclass
class ExperimentLogger:
    """Unified logger for console and optional Weights & Biases logging."""

    mode: str = "console"
    train_log_interval: int = 1
    console: bool = True

    def __post_init__(self) -> None:
        """Validate mode and initialize optional W&B client."""
        valid_modes = {"none", "console", "wandb", "both"}
        self.mode = str(self.mode).lower()
        if self.mode not in valid_modes:
            raise ValueError("`mode` must be one of {'none', 'console', 'wandb', 'both'}." )
        self.train_log_interval = max(int(self.train_log_interval), 1)
        self._wandb = None
        if self.mode in {"wandb", "both"}:
            try:
                import wandb  # type: ignore

                self._wandb = wandb
            except Exception as exc:  # pragma: no cover
                if self.mode == "wandb":
                    raise RuntimeError("W&B logging is enabled but wandb is not available.") from exc
                self.mode = "console"

    @classmethod
    def create(
        cls,
        *,
        use_wandb: bool = False,
        log_mode: str = "auto",
        train_log_interval: int = 1,
        console: bool = True,
    ) -> "ExperimentLogger":
        """Construct logger from CLI-style flags."""
        resolved_mode = log_mode.lower()
        if resolved_mode == "auto":
            resolved_mode = "wandb" if use_wandb else "console"
        return cls(mode=resolved_mode, train_log_interval=train_log_interval, console=console)

    def should_log_train(self, epoch: int) -> bool:
        """Return True when train metrics should be logged at this epoch."""
        return (epoch % self.train_log_interval) == 0

    def info(self, message: str) -> None:
        """Print an informational message when console logging is enabled."""
        if self.console and self.mode in {"console", "both", "wandb"}:
            print(message)

    def log(self, metrics: Mapping[str, Any], *, commit: bool = True) -> None:
        """Log a metrics dictionary to configured backends."""
        if self.mode == "none":
            return
        float_metrics = _to_float_metrics(metrics)
        if not float_metrics:
            return
        if self._wandb is not None and self.mode in {"wandb", "both"}:
            self._wandb.log(float_metrics, commit=commit)

