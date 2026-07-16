"""Reusable reset-time domain randomizers for ioailab tasks.

Each randomizer is task-agnostic IsaacLab event mechanics, organized by domain.
Reference a randomizer's ``apply`` classmethod as the ``func`` of an IsaacLab
``EventTermCfg`` (``mode="reset"``); ranges, asset names, and material/light
selections stay task-owned in the event ``params``.

    from ioailab.randomizers import ObjectPoseRandomizer
    EventTerm(func=ObjectPoseRandomizer.apply, mode="reset", params={...})
"""

from ioailab.randomizers.base import Randomizer
from ioailab.randomizers.camera import CameraPoseRandomizer
from ioailab.randomizers.lighting import DomeLightTextureRandomizer
from ioailab.randomizers.material import VisualMaterialRandomizer
from ioailab.randomizers.pose import (
    ObjectPoseRandomizer,
    ObjectSlotAssignmentRandomizer,
)

__all__ = [
    "CameraPoseRandomizer",
    "DomeLightTextureRandomizer",
    "ObjectPoseRandomizer",
    "ObjectSlotAssignmentRandomizer",
    "Randomizer",
    "VisualMaterialRandomizer",
]
