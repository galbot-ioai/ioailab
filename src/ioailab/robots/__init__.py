"""Robot component namespace for ioailab.

Shared robot-generic helpers live under ``ioailab.robots.common.actions`` and
``ioailab.robots.common.sensors``. Concrete robot bindings live under
``ioailab.robots.<robot>``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_G1_EXPORTS = {
    "G1_ACTION_JOINT_NAMES",
    "G1_LEG_DOF_ORDER",
    "G1_LEFT_ARM_DOF_ORDER",
    "G1_LEFT_GRIPPER_DOF_ORDER",
    "G1_RIGHT_ARM_DOF_ORDER",
    "G1_RIGHT_GRIPPER_DOF_ORDER",
    "GALBOT_G1_CFG",
    "is_galbot_g1_asset_available",
    "make_galbot_g1_articulation_cfg",
    "make_galbot_g1_manipulation_articulation_cfg",
    "resolve_galbot_g1_usd_path",
}

__all__ = sorted(_LAZY_G1_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily return G1 articulation exports."""

    if name not in _LAZY_G1_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("ioailab.robots.g1.articulation"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module attributes including lazy public exports."""

    return sorted((*globals(), *__all__))
