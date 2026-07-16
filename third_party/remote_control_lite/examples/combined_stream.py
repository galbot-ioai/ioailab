"""Combined frame example showing both arms and joysticks."""

import argparse
import queue
import time

from remote_control_lite import ArmSide, CombinedStreamDriver, QueueSink


def describe_joint(side: ArmSide, payload):
    joint = payload["joint"]["left" if side == ArmSide.Left else "right"]
    return f"seq={joint['sequence']} q0={joint['position'][0]:.3f}"


def describe_joystick(side: ArmSide, payload):
    stick = payload["joystick"]["left" if side == ArmSide.Left else "right"]
    axis = stick["axis"]
    return f"seq={stick['sequence']} axis=({axis['x']}, {axis['y']})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/galbotV1RemoteOperate")
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()

    sink = QueueSink(maxsize=16)
    driver = CombinedStreamDriver(port=args.port)
    driver.attach_sink(sink)

    with driver:
        deadline = time.time() + args.seconds
        try:
            while time.time() < deadline:
                try:
                    frame = sink.get(timeout=0.5)
                    left_joint = describe_joint(ArmSide.Left, frame)
                    right_joint = describe_joint(ArmSide.Right, frame)
                    left_stick = describe_joystick(ArmSide.Left, frame)
                    right_stick = describe_joystick(ArmSide.Right, frame)
                    print(
                        f"t={frame['timestamp']:.3f} | "
                        f"L({left_joint}, {left_stick}) | "
                        f"R({right_joint}, {right_stick})"
                    )
                except queue.Empty:
                    print("waiting for frame...")
        except KeyboardInterrupt:
            # exiting driver context will stop the stream and driver automaitically
            print("Interrupted by user, stopping stream.")


if __name__ == "__main__":
    main()
