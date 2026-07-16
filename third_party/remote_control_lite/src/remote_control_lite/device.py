"""Simplified driver implementations for the remote teleoperation arm."""

import logging
import math
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ._remote_arm_cpp import ArmSample, ArmSide, JoystickSample, RemoteArmDevice
from .sinks import BaseSink

LOGGER = logging.getLogger(__name__)

_JOINT_COUNT = 7
_DEFAULT_INIT = [0.0] * _JOINT_COUNT
# _DEFAULT_LEFT_SCALE = [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
# _DEFAULT_RIGHT_SCALE = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
_DEFAULT_LEFT_SCALE = [1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0]
_DEFAULT_RIGHT_SCALE = [1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0]

_BUTTON_COUNT = 9

_SideInput = Union[ArmSide, str]


def _side_name(side: ArmSide) -> str:
    return side.name.lower()


def _resolve_side(value: _SideInput) -> ArmSide:
    if isinstance(value, ArmSide):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"left", "l"}:
        return ArmSide.Left
    if normalized in {"right", "r"}:
        return ArmSide.Right
    raise ValueError("side must be 'left' or 'right'")


def _sample_to_dict(side: ArmSide, sample: ArmSample) -> Dict[str, object]:
    return {
        "side": _side_name(side),
        "sequence": int(sample.sequence),
        "timestamp": float(sample.timestamp),
        "position": list(sample.position),
        "velocity": list(sample.velocity),
        "torque": list(sample.torque),
    }


def _joystick_to_dict(side: ArmSide, sample: JoystickSample) -> Dict[str, object]:
    axis_x, axis_y = sample.axis
    trig_x, trig_y = sample.trigger
    return {
        "side": _side_name(side),
        "sequence": int(sample.sequence),
        "timestamp": float(sample.timestamp),
        "axis": {"x": int(axis_x), "y": int(axis_y)},
        "trigger": {"x": int(trig_x), "y": int(trig_y)},
        "buttons": [int(val) for val in sample.buttons],
    }


def _normalize_robot_name(value: str) -> str:
    value = value.strip()
    if not value:
        return "galbot_one_foxtrot"
    if value.startswith("galbot_one_"):
        return value
    if value.startswith("galbot_"):
        return value
    return f"galbot_one_{value}"


def _clamp(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def _joy_trigger_mapping(data: float, threshold: float, max_width: float) -> float:
    """Python port of remote_control::JoyTriggerMapping."""
    data = _clamp(data, 0.0, 1.0)
    if threshold <= 0.0:
        return _clamp(data * max_width, 0.0, max_width)
    if data <= threshold:
        return (data / threshold) * max_width
    return max_width


def _joy_axes_mapping(data: float, threshold: float) -> float:
    """Python port of remote_control::JoyAxesMapping."""
    data = _clamp(data, -1.0, 1.0)
    threshold = abs(threshold)
    if threshold >= 1.0:
        return 0.0
    ampl = (1.0 - threshold) * (1.0 - threshold)
    if ampl <= 0.0:
        return 0.0
    if data >= threshold:
        return ((data - threshold) * (data - threshold)) / ampl
    if data <= -threshold:
        return -((data + threshold) * (data + threshold)) / ampl
    return 0.0


def _angle_wrap(value: float) -> float:
    """Wrap angle into [-pi, pi] to align with remote_control."""
    two_pi = 2.0 * math.pi
    while value < -math.pi:
        value += two_pi
    while value > math.pi:
        value -= two_pi
    return value


class _ActionConstants:
    SOFT_EMERGENCY_STOP = -1
    SOFT_EMERGENCY_STOP_RESUME = 0
    LEFT_START_SUCTION_OR_CLOSE_GRIPPER_ACTION = 1
    LEFT_STOP_SUCTION_OR_OPEN_GRIPPER_ACTION = 2
    LEFT_SUCTION_TURN_RIGHT_ANGLE_ACTION = 3
    LEFT_SUCTION_TURN_ZERO_ACTION = 4
    RIGHT_START_SUCTION_OR_CLOSE_GRIPPER_ACTION = 5
    RIGHT_STOP_SUCTION_OR_OPEN_GRIPPER_ACTION = 6
    RIGHT_SUCTION_TURN_RIGHT_ANGLE_ACTION = 7
    RIGHT_SUCTION_TURN_ZERO_ACTION = 8
    CHASSIS_MOVE = 9
    CHASSIS_STOP = 10
    LEFT_GRIPPER_PAUSE = 11
    RIGHT_GRIPPER_PAUSE = 12
    LEFT_GRIPPER_ACTIVATE = 13
    RIGHT_GRIPPER_ACTIVATE = 14
    LEFT_ARM_SYNC = 15
    RIGHT_ARM_SYNC = 16
    LEFT_ARM_TAKEOVER_C = 17
    RIGHT_ARM_TAKEOVER_C = 18
    LEFT_ARM_TAKEOVER_X = 19
    RIGHT_ARM_TAKEOVER_X = 20
    LEG_DELTA_Z = 21
    LEG_DELTA_X = 22
    LEG_DELTA_YAW = 23
    HEAD_DELTA_YAW = 24
    HEAD_DELTA_PITCH = 25


class _ActionState:
    """State machine that emulates remote_control::ScaledDevice action_list outputs."""

    KEY_DOWN = 1
    KEY_UP = 2
    KEY_LONG = 3
    KEY_CLICK = 4

    LEFT_BUTTON_MAP = {
        0: "left_ctrl",
        1: "left_record",
        2: "left_R",
        3: "left_L",
        4: "left_arm_pause",
        5: "left_gripper_pause",
        6: "left_stop",
        7: "left_gripper_close",
        8: "left_gripper_open",
    }
    RIGHT_BUTTON_MAP = {
        0: "right_ctrl",
        1: "right_record",
        2: "right_L",
        3: "right_R",
        4: "right_arm_pause",
        5: "right_gripper_pause",
        6: "right_stop",
        7: "right_gripper_close",
        8: "right_gripper_open",
    }
    _GRIPPER_BUTTONS = {
        "left_gripper_close": ("left", True),
        "left_gripper_open": ("left", False),
        "right_gripper_close": ("right", True),
        "right_gripper_open": ("right", False),
    }

    def __init__(self, *, config: Optional[Dict[str, object]] = None, remote_ns: str = "remoter"):
        cfg = config or {}
        joy_cfg = cfg.get("joy", {})
        self.left_axes_x_threshold = float(joy_cfg.get("left_axes_x_threshold", 0.5))
        self.left_axes_y_threshold = float(joy_cfg.get("left_axes_y_threshold", 0.3))
        self.right_axes_x_threshold = float(joy_cfg.get("right_axes_x_threshold", 0.2))
        self.right_axes_y_threshold = float(joy_cfg.get("right_axes_y_threshold", 0.2))
        self.left_trigger_threshold = float(joy_cfg.get("left_trigger_threshold", 0.9))
        self.right_trigger_threshold = float(joy_cfg.get("right_trigger_threshold", 0.9))
        self.left_trigger_max_width = float(joy_cfg.get("left_trigger_max_width", 0.7))
        self.right_trigger_max_width = float(joy_cfg.get("right_trigger_max_width", 0.7))

        self.remote_ns = remote_ns
        self.axes: List[float] = [0.0] * 10
        self.axes[3] = self.left_trigger_max_width
        self.axes[4] = self.right_trigger_max_width

        # Motion toggles (mirrors ScaledDevice defaults)
        self.left_arm_sync = False
        self.right_arm_sync = False
        self.left_arm_takeover_c = False
        self.right_arm_takeover_c = False
        self.left_arm_takeover_x = False
        self.right_arm_takeover_x = False
        self.action_leg_dz = False
        self.action_leg_dx = False
        self.action_leg_dyaw = False
        self.action_head_dyaw = False
        self.action_head_dpitch = False
        self.l_gripper_pause = False
        self.r_gripper_pause = False
        self.left_gripper_close_long = False
        self.left_gripper_open_long = False
        self.right_gripper_close_long = False
        self.right_gripper_open_long = False
        self.chassis_move_translate = False
        self.chassis_move_rotate = False
        self.keyC_long_left = False
        self.keyC_long_right = False
        self.arm_record = False
        self.keyrecord_long_left = False
        self.keyrecord_long_right = False
        self.left_arm_catch = 0
        self.right_arm_catch = 0

        # Counters copied from remote_control implementation.
        self.chassis_stop_repeat = 0
        self.left_trigger_pub_cnt = 0
        self.right_trigger_pub_cnt = 0
        self.left_trigger_pause_repeat = 0
        self.right_trigger_pause_repeat = 0

        self._action_set: set[int] = set()

    def update_left_axis(self, raw_x: Optional[int], raw_y: Optional[int]) -> None:
        if raw_x is None or raw_y is None:
            return
        fb = _clamp(((float(raw_x) - 50.0) / 50.0) * -1.0, -1.0, 1.0)
        lr = _clamp(((50.0 - float(raw_y)) / 50.0) * -1.0, -1.0, 1.0)
        if not self.keyC_long_right:
            if abs(lr) < self.left_axes_y_threshold:
                self.axes[2] = 0.0
                self.chassis_move_rotate = False
            else:
                self.axes[2] = _joy_axes_mapping(lr, self.left_axes_y_threshold)
                self.chassis_move_rotate = True
            if abs(fb) < self.left_axes_x_threshold:
                self.axes[5] = 0.0
                self.action_leg_dz = False
            else:
                self.axes[5] = _joy_axes_mapping(fb, self.left_axes_x_threshold)
                self.action_leg_dz = True
            self.axes[6] = 0.0
            self.axes[7] = 0.0
            self.action_leg_dx = False
            self.action_leg_dyaw = False
        else:
            if abs(fb) < self.left_axes_x_threshold:
                self.axes[6] = 0.0
                self.action_leg_dx = False
            else:
                self.axes[6] = _joy_axes_mapping(fb, self.left_axes_x_threshold)
                self.action_leg_dx = True
            if abs(lr) < self.left_axes_y_threshold:
                self.axes[7] = 0.0
                self.action_leg_dyaw = False
            else:
                self.axes[7] = _joy_axes_mapping(lr, self.left_axes_y_threshold)
                self.action_leg_dyaw = True
            self.axes[5] = 0.0
            self.action_leg_dz = False
            self.chassis_move_rotate = False

    def update_right_axis(self, raw_x: Optional[int], raw_y: Optional[int]) -> None:
        if raw_x is None or raw_y is None:
            return
        fb = _clamp((float(raw_x) - 50.0) / 50.0, -1.0, 1.0)
        lr = _clamp((50.0 - float(raw_y)) / 50.0, -1.0, 1.0)
        if not self.keyC_long_left:
            if abs(fb) >= self.right_axes_x_threshold and abs(lr) >= self.right_axes_y_threshold:
                self.axes[0] = _joy_axes_mapping(fb, self.right_axes_x_threshold)
                self.axes[1] = _joy_axes_mapping(lr, self.right_axes_y_threshold)
                self.chassis_move_translate = True
            elif abs(fb) >= self.right_axes_x_threshold:
                self.axes[0] = _joy_axes_mapping(fb, self.right_axes_x_threshold)
                self.axes[1] = 0.0
                self.chassis_move_translate = True
            elif abs(lr) >= self.right_axes_y_threshold:
                self.axes[0] = 0.0
                self.axes[1] = _joy_axes_mapping(lr, self.right_axes_y_threshold)
                self.chassis_move_translate = True
            else:
                self.axes[0] = 0.0
                self.axes[1] = 0.0
                self.chassis_move_translate = False
            self.axes[8] = 0.0
            self.axes[9] = 0.0
            self.action_head_dyaw = False
            self.action_head_dpitch = False
        else:
            if abs(lr) < self.right_axes_y_threshold:
                self.axes[8] = 0.0
                self.action_head_dyaw = False
            else:
                self.axes[8] = _joy_axes_mapping(lr, self.right_axes_y_threshold)
                self.action_head_dyaw = True
            if abs(fb) < self.right_axes_x_threshold:
                self.axes[9] = 0.0
                self.action_head_dpitch = False
            else:
                self.axes[9] = _joy_axes_mapping(-fb, self.right_axes_x_threshold)
                self.action_head_dpitch = True
            self.chassis_move_translate = False

    def update_trigger(self, side: ArmSide, raw_value: Optional[int]) -> None:
        if raw_value is None:
            return
        width_percent = 1.0 - (_clamp(float(raw_value), 0.0, 100.0) * 0.01)
        if side == ArmSide.Left:
            self.axes[3] = _joy_trigger_mapping(
                width_percent, self.left_trigger_threshold, self.left_trigger_max_width
            )
        else:
            self.axes[4] = _joy_trigger_mapping(
                width_percent, self.right_trigger_threshold, self.right_trigger_max_width
            )

    def _toggle_takeover(self, name: str, enable: bool) -> None:
        if name == "left_L":
            self.left_arm_takeover_x = enable
        elif name == "left_R":
            self.left_arm_takeover_c = enable
        elif name == "right_L":
            self.right_arm_takeover_c = enable
        elif name == "right_R":
            self.right_arm_takeover_x = enable

    def _handle_gripper_button(self, name: str, status: int) -> bool:
        info = self._GRIPPER_BUTTONS.get(name)
        if not info:
            return False
        side, closing = info
        attr = f"{side}_gripper_{'close' if closing else 'open'}_long"
        if status == self.KEY_DOWN:
            setattr(self, attr, False)
            return True
        if status == self.KEY_LONG:
            setattr(self, attr, True)
            return True
        if status in {self.KEY_UP, self.KEY_CLICK}:
            was_long = getattr(self, attr)
            setattr(self, attr, False)
            if side == "left":
                if was_long:
                    self._action_set.add(_ActionConstants.LEFT_GRIPPER_PAUSE)
                elif closing:
                    self._action_set.add(_ActionConstants.LEFT_START_SUCTION_OR_CLOSE_GRIPPER_ACTION)
                else:
                    self._action_set.add(_ActionConstants.LEFT_STOP_SUCTION_OR_OPEN_GRIPPER_ACTION)
            else:
                if was_long:
                    self._action_set.add(_ActionConstants.RIGHT_GRIPPER_PAUSE)
                elif closing:
                    self._action_set.add(_ActionConstants.RIGHT_START_SUCTION_OR_CLOSE_GRIPPER_ACTION)
                else:
                    self._action_set.add(_ActionConstants.RIGHT_STOP_SUCTION_OR_OPEN_GRIPPER_ACTION)
            return True
        return True

    def handle_button_event(self, name: str, status: int) -> None:
        if name is None:
            return
        if self._handle_gripper_button(name, status):
            return
        if status == self.KEY_DOWN:
            if name in {"left_stop", "right_stop"}:
                self._action_set.add(_ActionConstants.SOFT_EMERGENCY_STOP)
            elif name == "left_arm_pause":
                self.left_arm_sync = not self.left_arm_sync
                self.left_arm_takeover_c = False
                self.left_arm_takeover_x = False
            elif name == "right_arm_pause":
                self.right_arm_sync = not self.right_arm_sync
                self.right_arm_takeover_c = False
                self.right_arm_takeover_x = False
            elif name == "right_gripper_pause":
                self.r_gripper_pause = not self.r_gripper_pause
            elif name == "left_gripper_pause":
                self.l_gripper_pause = not self.l_gripper_pause
            elif name == "left_ctrl":
                self.keyC_long_left = True
            elif name == "right_ctrl":
                self.keyC_long_right = True
            else:
                self._toggle_takeover(name, True)
        elif status == self.KEY_LONG:
            if name == "left_ctrl":
                self.keyC_long_left = True
            elif name == "right_ctrl":
                self.keyC_long_right = True
            elif name == "left_record":
                if not self.arm_record and not self.keyrecord_long_left:
                    self.keyrecord_long_left = True
                    self.arm_record = True
                elif self.arm_record and not self.keyrecord_long_left:
                    self.keyrecord_long_left = True
                    self.arm_record = False
            elif name == "right_record":
                if not self.arm_record and not self.keyrecord_long_right:
                    self.keyrecord_long_right = True
                    self.arm_record = True
                elif self.arm_record and not self.keyrecord_long_right:
                    self.keyrecord_long_right = True
                    self.arm_record = False
            else:
                self._toggle_takeover(name, True)
        elif status == self.KEY_UP:
            if name in {"left_stop", "right_stop"}:
                self._action_set.add(_ActionConstants.SOFT_EMERGENCY_STOP_RESUME)
            elif name == "left_ctrl":
                self.keyC_long_left = False
            elif name == "right_ctrl":
                self.keyC_long_right = False
            elif name == "left_record":
                if not self.keyrecord_long_left:
                    self.left_arm_catch += 1
                self.keyrecord_long_left = False
                self.arm_record = False
            elif name == "right_record":
                if not self.keyrecord_long_right:
                    self.right_arm_catch -= 1
                self.keyrecord_long_right = False
                self.arm_record = False
            else:
                self._toggle_takeover(name, False)

    def consume_buttons(self, timestamp: float) -> Dict[str, object]:
        if self.chassis_move_translate or self.chassis_move_rotate:
            self._action_set.add(_ActionConstants.CHASSIS_MOVE)
            self.chassis_stop_repeat = 0
        else:
            self.chassis_stop_repeat += 1
            if self.chassis_stop_repeat < 10:
                self._action_set.add(_ActionConstants.CHASSIS_STOP)
                self.axes[0] = 0.0
                self.axes[1] = 0.0
                self.axes[2] = 0.0
            elif self.chassis_stop_repeat > 100_000:
                self.chassis_stop_repeat = 10

        if self.left_arm_sync or self.left_arm_takeover_c or self.left_arm_takeover_x:
            if not self.l_gripper_pause:
                self.left_trigger_pub_cnt += 1
                if self.left_trigger_pub_cnt >= 5:
                    self._action_set.add(_ActionConstants.LEFT_GRIPPER_ACTIVATE)
                    self.left_trigger_pub_cnt = 0
                self.left_trigger_pause_repeat = 0
            else:
                self.left_trigger_pause_repeat += 1
                if self.left_trigger_pause_repeat < 10:
                    self._action_set.add(_ActionConstants.LEFT_GRIPPER_PAUSE)
                elif self.left_trigger_pause_repeat > 100_000:
                    self.left_trigger_pause_repeat = 10
        else:
            self.left_trigger_pub_cnt = 0
            self.left_trigger_pause_repeat = 0

        if self.right_arm_sync or self.right_arm_takeover_c or self.right_arm_takeover_x:
            if not self.r_gripper_pause:
                self.right_trigger_pub_cnt += 1
                if self.right_trigger_pub_cnt >= 5:
                    self._action_set.add(_ActionConstants.RIGHT_GRIPPER_ACTIVATE)
                    self.right_trigger_pub_cnt = 0
                self.right_trigger_pause_repeat = 0
            else:
                self.right_trigger_pause_repeat += 1
                if self.right_trigger_pause_repeat < 10:
                    self._action_set.add(_ActionConstants.RIGHT_GRIPPER_PAUSE)
                elif self.right_trigger_pause_repeat > 100_000:
                    self.right_trigger_pause_repeat = 10
        else:
            self.right_trigger_pub_cnt = 0
            self.right_trigger_pause_repeat = 0

        if self.left_arm_sync:
            self._action_set.add(_ActionConstants.LEFT_ARM_SYNC)
        if self.right_arm_sync:
            self._action_set.add(_ActionConstants.RIGHT_ARM_SYNC)
        if self.left_arm_takeover_c:
            self._action_set.add(_ActionConstants.LEFT_ARM_TAKEOVER_C)
        if self.right_arm_takeover_c:
            self._action_set.add(_ActionConstants.RIGHT_ARM_TAKEOVER_C)
        if self.left_arm_takeover_x:
            self._action_set.add(_ActionConstants.LEFT_ARM_TAKEOVER_X)
        if self.right_arm_takeover_x:
            self._action_set.add(_ActionConstants.RIGHT_ARM_TAKEOVER_X)
        if self.action_leg_dz:
            self._action_set.add(_ActionConstants.LEG_DELTA_Z)
        if self.action_leg_dx:
            self._action_set.add(_ActionConstants.LEG_DELTA_X)
        if self.action_leg_dyaw:
            self._action_set.add(_ActionConstants.LEG_DELTA_YAW)
        if self.action_head_dyaw:
            self._action_set.add(_ActionConstants.HEAD_DELTA_YAW)
        if self.action_head_dpitch:
            self._action_set.add(_ActionConstants.HEAD_DELTA_PITCH)

        buttons = sorted(self._action_set)
        frame_id = "takeover" if buttons else "normal"
        self._action_set.clear()
        return {
            "header": {"stamp": timestamp, "frame_id": frame_id},
            "axes": list(self.axes),
            "buttons": buttons,
        }


class _JointStateBuilder:
    """Helper that formats joint samples to mimic remote_control topic layout."""

    def __init__(self, remote_ns: str = "remoter"):
        self.remote_ns = remote_ns
        self.left_joint_names = [f"left_arm_joint{i + 1}" for i in range(_JOINT_COUNT)]
        self.right_joint_names = [f"right_arm_joint{i + 1}" for i in range(_JOINT_COUNT)]

    def build(
        self, left_sample: Dict[str, object], right_sample: Dict[str, object]
    ) -> Dict[str, Dict[str, object]]:
        return {
            "left_arm": self._format_side(left_sample, "left_arm", self.left_joint_names),
            "right_arm": self._format_side(right_sample, "right_arm", self.right_joint_names),
        }

    def _format_side(
        self, sample: Dict[str, object], label: str, joint_names: Sequence[str]
    ) -> Dict[str, object]:
        positions = [_angle_wrap(float(val)) for val in sample["position"]]
        velocities = [float(val) for val in sample["velocity"]]
        efforts = [float(val) for val in sample["torque"]]
        return {
            "header": {
                "stamp": float(sample["timestamp"]),
                "frame_id": f"{self.remote_ns}/{label}",
            },
            "name": list(joint_names),
            "position": positions,
            "velocity": velocities,
            "effort": efforts,
        }


class BaseArmDriver:
    """Common lifecycle for all drivers (single sink allowed)."""

    def __init__(
        self,
        *,
        port: str = "/dev/galbotV1RemoteOperate",
        poll_interval: float = 0.01,
    ):
        self._port = port
        self._poll_interval = float(poll_interval)
        self._device = RemoteArmDevice()
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sink: Optional[BaseSink] = None
        self._lock = threading.Lock()

        self._device.configure(
            _DEFAULT_INIT,
            _DEFAULT_INIT,
            _DEFAULT_LEFT_SCALE,
            _DEFAULT_RIGHT_SCALE,
        )

    def attach_sink(self, sink: BaseSink) -> None:
        if sink is None:
            raise ValueError("sink cannot be None")
        if not isinstance(sink, BaseSink):
            raise TypeError("sink must inherit BaseSink")
        with self._lock:
            if self._sink is not None:
                raise RuntimeError("sink already attached")
            self._sink = sink
            sink.open(self)

    def detach_sink(self) -> None:
        with self._lock:
            if self._sink is None:
                return
            self._sink.close()
            self._sink = None

    def start(self) -> None:
        if self._running.is_set():
            return
        self._device.start(self._port)
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join()
            self._thread = None
        self._device.stop()

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.detach_sink()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _loop(self) -> None:
        try:
            while self._running.is_set():
                updated = self._poll_once()
                if not updated:
                    time.sleep(self._poll_interval)
        except Exception as exc:
            LOGGER.exception("driver loop terminated: %s", exc)
            raise

    def _poll_once(self) -> bool:
        raise NotImplementedError

    def _push(self, payload: Dict[str, object]) -> None:
        sink = self._sink
        if sink is None:
            return
        try:
            sink.push(payload)
        except Exception as exc:
            LOGGER.debug("sink push failed: %s", exc)


class SingleStreamDriver(BaseArmDriver):
    """Streams a single joint or joystick sample for one side."""

    def __init__(
        self,
        *,
        side: _SideInput,
        stream: str = "joint",
        port: str = "/dev/galbotV1RemoteOperate",
        poll_interval: float = 0.01,
    ):
        super().__init__(port=port, poll_interval=poll_interval)
        if stream not in {"joint", "joystick"}:
            raise ValueError("stream must be 'joint' or 'joystick'")
        self._side = _resolve_side(side)
        self._stream = stream
        self._last_sequence = 0

    def _poll_once(self) -> bool:
        if self._stream == "joint":
            if not self._device.has_sample(self._side):
                return False
            sample = self._device.latest_sample(self._side)
            if sample.sequence == 0 or sample.sequence == self._last_sequence:
                return False
            self._last_sequence = int(sample.sequence)
            payload = _sample_to_dict(self._side, sample)
        else:
            if not self._device.has_joystick(self._side):
                return False
            sample = self._device.latest_joystick(self._side)
            if sample.sequence == 0 or sample.sequence == self._last_sequence:
                return False
            self._last_sequence = int(sample.sequence)
            payload = _joystick_to_dict(self._side, sample)
        self._push(payload)
        return True


class CombinedStreamDriver(BaseArmDriver):
    """Streams a combined frame containing all joints and joysticks."""

    def __init__(
        self,
        *,
        port: str = "/dev/galbotV1RemoteOperate",
        poll_interval: float = 0.01,
    ):
        super().__init__(port=port, poll_interval=poll_interval)
        self._joint_last = {ArmSide.Left: 0, ArmSide.Right: 0}
        self._stick_last = {ArmSide.Left: 0, ArmSide.Right: 0}
        self._latest_joint: Dict[ArmSide, Dict[str, object]] = {}
        self._latest_stick: Dict[ArmSide, Dict[str, object]] = {}

    def _poll_once(self) -> bool:
        updated = False
        for side in (ArmSide.Left, ArmSide.Right):
            if self._device.has_sample(side):
                sample = self._device.latest_sample(side)
                if sample.sequence and sample.sequence != self._joint_last[side]:
                    self._joint_last[side] = int(sample.sequence)
                    self._latest_joint[side] = _sample_to_dict(side, sample)
                    updated = True
            if self._device.has_joystick(side):
                sample = self._device.latest_joystick(side)
                if sample.sequence and sample.sequence != self._stick_last[side]:
                    self._stick_last[side] = int(sample.sequence)
                    self._latest_stick[side] = _joystick_to_dict(side, sample)
                    updated = True

        if not updated:
            return False

        if len(self._latest_joint) < 2 or len(self._latest_stick) < 2:
            return False

        joints = {side: self._latest_joint[side] for side in (ArmSide.Left, ArmSide.Right)}
        sticks = {side: self._latest_stick[side] for side in (ArmSide.Left, ArmSide.Right)}
        timestamps = [joints[ArmSide.Left]["timestamp"], joints[ArmSide.Right]["timestamp"]]
        timestamps.extend(
            [sticks[ArmSide.Left]["timestamp"], sticks[ArmSide.Right]["timestamp"]]
        )
        skew = max(timestamps) - min(timestamps)
        frame = {
            "timestamp": max(timestamps),
            "skew": skew,
            "joint": {
                "left": joints[ArmSide.Left],
                "right": joints[ArmSide.Right],
            },
            "joystick": {
                "left": sticks[ArmSide.Left],
                "right": sticks[ArmSide.Right],
            },
        }
        self._push(frame)
        return True


class GalbotDriver(CombinedStreamDriver):
    """Produces remote_control-compatible frames with action_list and joint_states."""

    # TODO @Haozhe Chen: some button on joystick is not handled, should be covered.
    #   check details in https://owm6ymi5v9b.feishu.cn/wiki/UX7PwC0CJiNERjkXKlLc4Hllnng
    def __init__(
        self,
        *,
        robot: str = "foxtrot",
        remote_ns: str = "remoter",
        action_config: Optional[Dict[str, object]] = None,
        port: str = "/dev/galbotV1RemoteOperate",
        poll_interval: float = 0.01,
    ):
        super().__init__(port=port, poll_interval=poll_interval)
        self._robot_name = _normalize_robot_name(robot)
        self._remote_ns = remote_ns.strip() or "remoter"
        self._action_state = _ActionState(config=action_config, remote_ns=self._remote_ns)
        self._joint_builder = _JointStateBuilder(remote_ns=self._remote_ns)
        self._button_states: Dict[ArmSide, List[int]] = {
            ArmSide.Left: [_ActionState.KEY_UP] * _BUTTON_COUNT,
            ArmSide.Right: [_ActionState.KEY_UP] * _BUTTON_COUNT,
        }
        self._side_button_maps: Dict[ArmSide, Dict[int, Optional[str]]] = {
            ArmSide.Left: _ActionState.LEFT_BUTTON_MAP,
            ArmSide.Right: _ActionState.RIGHT_BUTTON_MAP,
        }
        self._button_state: List[int] = [0] * (_BUTTON_COUNT * 2)

    def _process_stick(self, side: ArmSide, payload: Dict[str, object]) -> None:
        axis = payload.get("axis", {})
        trigger = payload.get("trigger", {})
        buttons = payload.get("buttons", [])
        raw_x = axis.get("x")
        raw_y = axis.get("y")
        trig_x = trigger.get("x")
        if side == ArmSide.Left:
            self._action_state.update_left_axis(raw_x, raw_y)
            self._action_state.update_trigger(ArmSide.Left, trig_x)
        else:
            self._action_state.update_right_axis(raw_x, raw_y)
            self._action_state.update_trigger(ArmSide.Right, trig_x)
        self._handle_button_changes(side, buttons)

    def _handle_button_changes(self, side: ArmSide, buttons: Sequence[int]) -> None:
        if not buttons:
            return
        prev = self._button_states[side]
        mapping = self._side_button_maps[side]
        offset = 0 if side == ArmSide.Left else _BUTTON_COUNT
        for idx in range(min(len(buttons), _BUTTON_COUNT)):
            incoming = int(buttons[idx])
            self._button_state[offset + idx] = incoming
            if incoming == prev[idx]:
                continue
            name = mapping.get(idx)
            self._action_state.handle_button_event(name, incoming)
            prev[idx] = incoming

    def _poll_once(self) -> bool:
        updated = False
        for side in (ArmSide.Left, ArmSide.Right):
            if self._device.has_sample(side):
                sample = self._device.latest_sample(side)
                if sample.sequence and sample.sequence != self._joint_last[side]:
                    self._joint_last[side] = int(sample.sequence)
                    self._latest_joint[side] = _sample_to_dict(side, sample)
                    updated = True
            if self._device.has_joystick(side):
                sample = self._device.latest_joystick(side)
                if sample.sequence and sample.sequence != self._stick_last[side]:
                    self._stick_last[side] = int(sample.sequence)
                    stick_payload = _joystick_to_dict(side, sample)
                    self._latest_stick[side] = stick_payload
                    self._process_stick(side, stick_payload)
                    updated = True

        if not updated:
            return False
        if len(self._latest_joint) < 2 or len(self._latest_stick) < 2:
            return False

        left_joint = self._latest_joint[ArmSide.Left]
        right_joint = self._latest_joint[ArmSide.Right]
        left_stick = self._latest_stick[ArmSide.Left]
        right_stick = self._latest_stick[ArmSide.Right]

        timestamps: List[float] = [
            float(left_joint["timestamp"]),
            float(right_joint["timestamp"]),
            float(left_stick["timestamp"]),
            float(right_stick["timestamp"]),
        ]
        timestamp = max(timestamps)
        skew = max(timestamps) - min(timestamps)

        action_list = self._action_state.consume_buttons(timestamp)
        joint_states = self._joint_builder.build(left_joint, right_joint)
        # include robot metadata for downstream consumers.
        joint_states["robot"] = self._robot_name

        frame = {
            "timestamp": timestamp,
            "skew": skew,
            "action_list": action_list,
            "joint_states": joint_states,
            "button_state": list(self._button_state),
        }
        self._push(frame)
        return True
