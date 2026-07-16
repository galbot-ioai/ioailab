import ctypes
import os
from pathlib import Path

__version__ = "0.1.0"

# Preload bundled native libs to avoid rpath issues in editable installs.
_PKG_DIR = Path(__file__).resolve().parent
_LIBS_DIR = _PKG_DIR / "libs"

_PRELOAD_ORDER = [
    "libgalbotLog.so",
    "libgalbotSerialPort.so",
    "libgalbotDataProtocol.so",
    "libgalbotSystem.so",
    "libgalbotRemoteOperate.so",
]

if _LIBS_DIR.is_dir():
    for _name in _PRELOAD_ORDER:
        _p = _LIBS_DIR / _name
        if _p.exists():
            try:
                ctypes.CDLL(str(_p), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                # Defer failure to extension import for clearer stacktrace
                pass

from ._remote_arm_cpp import ArmSample, ArmSide, JoystickSample
from .device import BaseArmDriver, CombinedStreamDriver, GalbotDriver, SingleStreamDriver
from .queue_mp import QueueSinkClient, QueueSinkServer
from .sinks import BaseSink, QueueSink, PrintSink, SharedMemorySink, UDPSink

__all__ = [
    "ArmSample",
    "ArmSide",
    "JoystickSample",
    "BaseArmDriver",
    "SingleStreamDriver",
    "CombinedStreamDriver",
    "GalbotDriver",
    "BaseSink",
    "QueueSink",
    "QueueSinkServer",
    "QueueSinkClient",
    "PrintSink",
    "SharedMemorySink",
    "UDPSink",
    "__version__",
]
