"""Export sorting-object meshes for FoundationPose.

Generates 4 watertight meshes (red_cube, blue_cuboid, yellow_cylinder,
green_cylinder) matching the sort-to-shelf scene geometry and saves them to
``data/foundationpose_bridge/sort_to_shelf/request/meshes/`` so the
FoundationPose server can find them by object name.

Run this script inside the dev container:

    python examples/vision_baseline/scripts/export_sorting_meshes.py

The FoundationPose server selects the mesh matching the requested object at
inference time. Regenerate only when object geometry changes in
src/ioailab/tasks/sort_to_shelf/scene.py.

Origin convention: bottom center (Z=0 at base, XY centered).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_OUTPUT_DIR = "data/foundationpose_bridge/sort_to_shelf/request/meshes"


# Hardcoded sorting object specs matching SORTING_OBJECT_SPECS in scene.py
SORTING_OBJECTS = {
    "red_cube": {
        "size": (0.06, 0.06, 0.07),  # (x, y, z) in meters
        "color": (0.85, 0.18, 0.12),  # RGB [0, 1]
        "type": "box",
    },
    "blue_cuboid": {
        "size": (0.05, 0.05, 0.12),
        "color": (0.08, 0.22, 0.85),
        "type": "box",
    },
    "yellow_cylinder": {
        "size": (0.05, 0.05, 0.12),  # (x, y, z) where radius=x/2, height=z
        "color": (0.95, 0.78, 0.08),
        "type": "cylinder",
    },
    "green_cylinder": {
        "size": (0.06, 0.06, 0.08),
        "color": (0.15, 0.60, 0.22),
        "type": "cylinder",
    },
}


def main(argv: list[str] | None = None) -> None:
    """Export all four sorting-object meshes."""

    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, spec in SORTING_OBJECTS.items():
        mesh_path = output_dir / f"{name}.obj"
        if spec["type"] == "cylinder":
            export_cylinder_mesh(
                mesh_path,
                radius=spec["size"][0] / 2.0,
                height=spec["size"][2],
                color=spec["color"],
            )
        else:
            export_box_mesh(mesh_path, extents=spec["size"], color=spec["color"])
        print(f"[export] {mesh_path}")

    print(f"\n[export] Exported 4 meshes to {output_dir}")
    print("[export] The FoundationPose server selects per-object by name.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory to write .obj files. Must match the request/meshes/ "
            f"path expected by the FoundationPose server. Defaults to {DEFAULT_OUTPUT_DIR}."
        ),
    )
    return parser.parse_args(argv)


def export_box_mesh(
    output_path: str | Path,
    extents: tuple[float, float, float],
    *,
    color: tuple[float, float, float] = (0.85, 0.18, 0.12),
) -> Path:
    """Write a box mesh with origin at bottom center to ``output_path``."""

    import trimesh

    mesh = trimesh.creation.box(extents=tuple(float(e) for e in extents))
    # Shift mesh so origin is at bottom center (default origin is geometric center)
    mesh.vertices[:, 2] += extents[2] / 2.0
    rgba = np.array(
        [*(int(round(channel * 255)) for channel in color), 255], dtype=np.uint8
    )
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba)
    dst = Path(output_path)
    mesh.export(str(dst))
    return dst


def export_cylinder_mesh(
    output_path: str | Path,
    radius: float,
    height: float,
    *,
    color: tuple[float, float, float] = (0.85, 0.85, 0.12),
    sections: int = 32,
) -> Path:
    """Write a cylinder mesh with origin at bottom center to ``output_path``."""

    import trimesh

    mesh = trimesh.creation.cylinder(
        radius=float(radius), height=float(height), sections=sections
    )
    # Trimesh cylinder default origin is at geometric center (Z=0 at half height)
    # Shift so origin is at bottom center (Z=0 at base)
    mesh.vertices[:, 2] += height / 2.0
    rgba = np.array(
        [*(int(round(channel * 255)) for channel in color), 255], dtype=np.uint8
    )
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba)
    dst = Path(output_path)
    mesh.export(str(dst))
    return dst


if __name__ == "__main__":
    main()
