from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

ROOT = Path(__file__).parent.resolve()
THIRD_PARTY_INCLUDE = ROOT / "third_party" / "include"
LIB_DIR = ROOT / "src" / "remote_control_lite" / "libs"

ext_modules = [
    Pybind11Extension(
        "remote_control_lite._remote_arm_cpp",
        sources=[
            "cpp/src/device.cpp",
            "cpp/src/bindings.cpp",
        ],
        include_dirs=[
            "cpp/include",
            str(THIRD_PARTY_INCLUDE),
        ],
        library_dirs=[
            str(LIB_DIR),
        ],
        libraries=[
            "galbotRemoteOperate",
            "galbotSystem",
            "galbotDataProtocol",
            "galbotSerialPort",
            "galbotLog",
        ],
        extra_compile_args=["-std=c++17"],
        extra_link_args=["-Wl,-rpath,$ORIGIN/libs"],
    ),
]

setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
