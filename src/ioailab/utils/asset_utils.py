"""Project-local asset lookup helpers for ioailab examples and configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
G1_DESCRIPTION_ASSET_ROOT = PROJECT_ROOT / "assets" / "galbot_one_golf_description"
G1_USD_PATH = G1_DESCRIPTION_ASSET_ROOT / "usd" / "galbot_one_golf.usda"
G1_URDF_PATH = G1_DESCRIPTION_ASSET_ROOT / "urdf" / "galbot_one_golf.urdf"
G1_GENERATED_ASSET_ROOT = (
    PROJECT_ROOT / "assets" / "generated" / "galbot_one_golf_description"
)
G1_MOBILE_BASE_URDF_PATH = (
    G1_GENERATED_ASSET_ROOT / "urdf" / "galbot_one_golf_mobile_base.urdf"
)
MATERIAL_ASSET_ROOT = PROJECT_ROOT / "assets" / "materials"
HDRI_ASSET_ROOT = PROJECT_ROOT / "assets" / "hdris"
OBJECT_ASSET_ROOT = PROJECT_ROOT / "assets" / "objects"


@dataclass(frozen=True)
class RobotAsset:
    """Static metadata for a robot asset expected under ``assets/``."""

    name: str
    display_name: str
    usd_path: Path
    urdf_path: Path


@dataclass(frozen=True)
class ObjectAsset:
    """Static metadata for a small object asset expected under ``assets/``."""

    name: str
    display_name: str
    variant: str
    root_path: Path
    usd_path: Path
    visual_glb_path: Path
    collision_glb_path: Path
    metadata_path: Path
    points_info_path: Path


ROBOT_ASSETS: dict[str, RobotAsset] = {
    "galbot_g1": RobotAsset(
        name="galbot_g1",
        display_name="Galbot G1",
        usd_path=G1_USD_PATH,
        urdf_path=G1_URDF_PATH,
    ),
}


def require_file(path: Path, *, asset_type: str, name: str) -> None:
    """Raise a repo-relative error when a required asset file is missing."""

    if not path.is_file():
        relative_path = path.relative_to(PROJECT_ROOT)
        raise FileNotFoundError(
            f"{asset_type} asset '{name}' is missing at {relative_path}."
        )


def get_robot_asset(name: str) -> RobotAsset:
    """Return static local metadata for a robot asset."""

    try:
        return ROBOT_ASSETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown robot asset '{name}'.") from exc


def get_robot_usd_path(name: str, *, required: bool = True) -> Path:
    """Return the local USD path for a robot asset."""

    asset = get_robot_asset(name)
    if required:
        require_file(asset.usd_path, asset_type="Robot USD", name=name)
    return asset.usd_path


def get_robot_urdf_path(name: str, *, required: bool = True) -> Path:
    """Return the local canonical URDF path for a robot asset."""

    asset = get_robot_asset(name)
    if required:
        require_file(asset.urdf_path, asset_type="Robot URDF", name=name)
    return asset.urdf_path


def list_visual_material_paths(
    *,
    categories: tuple[str, ...] | None = None,
    required: bool = False,
) -> tuple[Path, ...]:
    """Return local MDL visual material paths under ``assets/materials``."""

    if not MATERIAL_ASSET_ROOT.is_dir():
        paths: tuple[Path, ...] = ()
    else:
        material_paths = sorted(MATERIAL_ASSET_ROOT.rglob("*.mdl"))
        if categories is not None:
            category_names = set(categories)
            material_paths = [
                path
                for path in material_paths
                if any(parent.name in category_names for parent in path.parents)
            ]
        paths = tuple(material_paths)

    if required and not paths:
        relative_path = MATERIAL_ASSET_ROOT.relative_to(PROJECT_ROOT)
        raise FileNotFoundError(
            f"No visual material assets found under {relative_path}."
        )
    return paths


def list_hdri_paths(*, required: bool = False) -> tuple[Path, ...]:
    """Return local HDRI texture paths under ``assets/hdris``."""

    hdri_suffixes = {".exr", ".hdr"}
    if not HDRI_ASSET_ROOT.is_dir():
        paths: tuple[Path, ...] = ()
    else:
        paths = tuple(
            sorted(
                path
                for path in HDRI_ASSET_ROOT.rglob("*")
                if path.suffix.lower() in hdri_suffixes
            )
        )

    if required and not paths:
        relative_path = HDRI_ASSET_ROOT.relative_to(PROJECT_ROOT)
        raise FileNotFoundError(f"No HDRI texture assets found under {relative_path}.")
    return paths


def list_object_asset_names() -> tuple[str, ...]:
    """Return object asset directory names available under ``assets/objects``."""

    if not OBJECT_ASSET_ROOT.is_dir():
        return ()
    return tuple(
        sorted(path.name for path in OBJECT_ASSET_ROOT.iterdir() if path.is_dir())
    )


def _object_variant_suffix(variant: str) -> str:
    """Return the numeric suffix used by ``model_data{suffix}.json`` files."""

    if variant.startswith("base"):
        return variant.removeprefix("base")
    return variant


def _discover_object_variant(root_path: Path, *, name: str, required: bool) -> str:
    """Resolve the single selected USD variant for one object directory."""

    usd_dir = root_path / "usd"
    variants = (
        tuple(sorted(path.stem for path in usd_dir.glob("*.usd")))
        if usd_dir.is_dir()
        else ()
    )
    if len(variants) == 1:
        return variants[0]
    if not required:
        return "base0"
    if not variants:
        relative_path = usd_dir.relative_to(PROJECT_ROOT)
        raise FileNotFoundError(
            f"Object asset '{name}' has no USD variants under {relative_path}."
        )
    raise ValueError(
        f"Object asset '{name}' has multiple USD variants {variants}; pass variant explicitly."
    )


def _make_object_asset(name: str, *, variant: str, root_path: Path) -> ObjectAsset:
    """Build object metadata from the standard ``assets/objects`` layout."""

    metadata_suffix = _object_variant_suffix(variant)
    return ObjectAsset(
        name=name,
        display_name=name.replace("_", " ").title(),
        variant=variant,
        root_path=root_path,
        usd_path=root_path / "usd" / f"{variant}.usd",
        visual_glb_path=root_path / "visual" / f"{variant}.glb",
        collision_glb_path=root_path / "collision" / f"{variant}.glb",
        metadata_path=root_path / f"model_data{metadata_suffix}.json",
        points_info_path=root_path / "points_info.json",
    )


def get_object_asset(
    name: str, *, variant: str | None = None, required: bool = True
) -> ObjectAsset:
    """Return local metadata for an object asset under ``assets/objects``."""

    root_path = OBJECT_ASSET_ROOT / name
    if required and not root_path.is_dir():
        relative_path = root_path.relative_to(PROJECT_ROOT)
        raise FileNotFoundError(f"Object asset '{name}' is missing at {relative_path}.")

    resolved_variant = variant or _discover_object_variant(
        root_path, name=name, required=required
    )
    asset = _make_object_asset(name, variant=resolved_variant, root_path=root_path)
    if required:
        for asset_type, path in (
            ("Object USD", asset.usd_path),
            ("Object visual GLB", asset.visual_glb_path),
            ("Object collision GLB", asset.collision_glb_path),
            ("Object metadata", asset.metadata_path),
            ("Object points info", asset.points_info_path),
        ):
            require_file(path, asset_type=asset_type, name=name)
    return asset


def get_object_usd_path(
    name: str, *, variant: str | None = None, required: bool = True
) -> Path:
    """Return the local USD path for an object asset."""

    asset = get_object_asset(name, variant=variant, required=required)
    if required:
        require_file(asset.usd_path, asset_type="Object USD", name=name)
    return asset.usd_path


def get_object_metadata_path(
    name: str, *, variant: str | None = None, required: bool = True
) -> Path:
    """Return the local metadata JSON path for an object asset."""

    asset = get_object_asset(name, variant=variant, required=required)
    if required:
        require_file(asset.metadata_path, asset_type="Object metadata", name=name)
    return asset.metadata_path


def get_object_points_info_path(name: str, *, required: bool = True) -> Path:
    """Return the local grasp/target points JSON path for an object asset."""

    asset = get_object_asset(name, required=required)
    if required:
        require_file(asset.points_info_path, asset_type="Object points info", name=name)
    return asset.points_info_path
