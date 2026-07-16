"""IsaacLab Mimic dataset expansion helpers."""

from __future__ import annotations

from types import ModuleType
import sys
from typing import Any

from ioailab.datasets.mimic.config import MimicCfg

__all__ = ["MimicCfg"]


class _CallableMimicModule(ModuleType):
    """Keep ``from ioailab.datasets import mimic`` callable after submodule import."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate module calls to the public dataset Mimic helper."""

        from ioailab.datasets.refs import mimic

        return mimic(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableMimicModule
