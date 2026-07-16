"""Base policy training contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class OptimizerCfg:
    """Optimizer settings shared by policy training adapters."""

    learning_rate: float = 1.0e-4
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None

    def __post_init__(self) -> None:
        """Validate optimizer settings."""

        if self.learning_rate <= 0.0:
            raise ValueError("OptimizerCfg.learning_rate must be greater than zero.")
        if self.weight_decay < 0.0:
            raise ValueError("OptimizerCfg.weight_decay must not be negative.")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError(
                "OptimizerCfg.grad_clip_norm must be greater than zero when provided."
            )


@dataclass(frozen=True, slots=True)
class PolicyTrainCfg:
    """Shared training configuration consumed by policy adapters."""

    output_dir: Path | str | None = None
    epochs: int | None = None
    batch_size: int | None = None
    seed: int | None = None
    device: str | None = None
    optimizer: OptimizerCfg = field(default_factory=OptimizerCfg)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize training configuration."""

        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.epochs is not None and self.epochs < 1:
            raise ValueError(
                "PolicyTrainCfg.epochs must be greater than zero when provided."
            )
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError(
                "PolicyTrainCfg.batch_size must be greater than zero when provided."
            )
        if self.seed is not None and self.seed < 0:
            raise ValueError("PolicyTrainCfg.seed must not be negative when provided.")
        if not isinstance(self.optimizer, OptimizerCfg):
            raise TypeError(
                "PolicyTrainCfg.optimizer must be an OptimizerCfg instance."
            )
        object.__setattr__(self, "metadata", dict(self.metadata))


class Policy:
    """Factory for supported ioailab policy backends."""

    @staticmethod
    def from_backend(backend: str) -> "Policy":
        """Return a policy adapter for ``backend``."""

        if backend != "robomimic_diffusion":
            raise ValueError(f"Unsupported policy backend: {backend}")

        from ioailab.agents.policy.backends.robomimic_diffusion import (
            RobomimicDiffusionPolicy,
        )

        return RobomimicDiffusionPolicy(backend=backend)
