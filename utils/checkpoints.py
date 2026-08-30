from __future__ import annotations

"""Checkpoint save/load helpers built on Orbax CheckpointManager."""

from pathlib import Path
from typing import Any, Optional

import orbax.checkpoint as ocp


def _manager_directory(ckpt_dir: str, prefix: str) -> Path:
    """Return checkpoint manager directory for a run prefix."""
    return Path(ckpt_dir) / prefix


def _create_manager(
    ckpt_dir: str,
    prefix: str,
    max_to_keep: int = 5,
) -> ocp.CheckpointManager:
    """Create a configured Orbax checkpoint manager."""
    manager_dir = _manager_directory(ckpt_dir, prefix)
    manager_dir.mkdir(parents=True, exist_ok=True)

    options = ocp.CheckpointManagerOptions(
        create=True,
        max_to_keep=max_to_keep,
    )
    # ✅ refactored API
    return ocp.CheckpointManager(manager_dir, options=options)


def save_train_state(
    state: Any,
    ckpt_dir: str,
    prefix: str,
    step: int,
    max_to_keep: int = 5,
) -> None:
    """Save training state at a given step."""
    step = int(step)
    with _create_manager(ckpt_dir, prefix, max_to_keep=max_to_keep) as manager:
        manager.save(step, args=ocp.args.StandardSave(state))
        manager.wait_until_finished()


def load_train_state(
    target_state: Any,
    ckpt_dir: str,
    prefix: str,
    step: Optional[int] = None,
    max_to_keep: int = 5,
) -> Any:
    """Restore training state from a checkpoint step (or latest when omitted)."""
    with _create_manager(ckpt_dir, prefix, max_to_keep=max_to_keep) as manager:
        if step is None:
            step = manager.latest_step()
            if step is None:
                raise FileNotFoundError(
                    f"No checkpoints found in '{_manager_directory(ckpt_dir, prefix)}'."
                )
        step = int(step)

        return manager.restore(step, args=ocp.args.StandardRestore(target_state))