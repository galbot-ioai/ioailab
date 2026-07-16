"""Policy agent taxonomy for checkpoint-backed action sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ioailab.agents.base import _ActionSourceAgent
from ioailab.agents.policy.checkpoint import PolicyCheckpoint

DEFAULT_POLICY_BACKEND = "robomimic_diffusion"


class PolicyAgent(_ActionSourceAgent):
    """Imitation-policy agent facade that emits task actions."""

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Any,
        *,
        backend: str = DEFAULT_POLICY_BACKEND,
        **metadata: Any,
    ) -> "PolicyAgent":
        """Create a runnable policy agent from a trained checkpoint."""

        from ioailab.agents.policy.backends.robomimic_diffusion import (
            _CheckpointActionSource,
            checkpoint_metadata_from_artifacts,
        )

        if isinstance(checkpoint, PolicyCheckpoint):
            checkpoint_ref = checkpoint
            if not checkpoint_ref.metadata:
                recovered = checkpoint_metadata_from_artifacts(checkpoint_ref.path)
                if recovered:
                    checkpoint_ref = PolicyCheckpoint(
                        path=checkpoint_ref.path,
                        backend=checkpoint_ref.backend,
                        metadata=recovered,
                    )
        else:
            checkpoint_path = (
                Path(checkpoint) if isinstance(checkpoint, str) else checkpoint
            )
            if not metadata:
                metadata = checkpoint_metadata_from_artifacts(checkpoint_path)
            checkpoint_ref = PolicyCheckpoint(
                path=checkpoint_path,
                backend=backend,
                metadata=metadata,
            )
        return cls(
            _CheckpointActionSource(checkpoint_ref),
            checkpoint_path=checkpoint_ref.path,
            backend=checkpoint_ref.backend,
            **checkpoint_ref.metadata,
        )
