# Sensors

Cameras are IsaacLab config inputs. A task scene cfg activates the robot-mounted
camera it needs; IsaacLab owns sensor creation, rendering, buffers, and runtime
tensor reads. `GalbotG1-PickCube-v0`, `GalbotG1-PickCube-Teleop-v0`, and
`GalbotG1-PickToShelf-v0` request camera rendering through task metadata.

## Activate a camera

Use the G1 sensor facade inside a task `env_cfg.py`:

```python
from ioailab.robots.g1 import g1


@configclass
class MySceneCfg(InteractiveSceneCfg):
    robot = ...
    left_wrist_rgb_camera = g1.sensors.camera("left_wrist")
    front_head_rgb_camera = g1.sensors.camera("front_head")
```

Valid G1 mounts: `front_head`, `left_wrist`, `right_wrist`. The cfgs reuse
calibrated mount transforms and intrinsics. There are no runtime camera-size or
data-type flags; for custom non-G1 cameras, use IsaacLab camera cfgs directly.

## Read at runtime

```python
from ioailab.envs import make_env

env = make_env("GalbotG1-PickCube-Teleop-v0", num_envs=1)
env.reset()
left_rgb = env.scene["left_wrist_rgb_camera"].data.output["rgb"]
head_rgb = env.scene["front_head_rgb_camera"].data.output["rgb"]
```

`GalbotG1-PickCube-Teleop-v0` owns the left-wrist and front-head RGB cameras and
records them through its MDP observation cfg during recorder-backed collection.
Use `ioailab.utils.rerun_utils` for optional Rerun viewer/URL helpers around
recorded streams, outside task sensor cfgs. When depth is enabled on a custom
camera, IsaacLab exposes it via keys such as `distance_to_image_plane`.
