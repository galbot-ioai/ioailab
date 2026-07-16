"""Combined frame example showing both arms and joysticks."""

import argparse
import queue
import time

from remote_control_lite import ArmSide, GalbotDriver, QueueSink


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/galbotV1RemoteOperate")
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()

    sink = QueueSink(maxsize=16)
    driver = GalbotDriver(port=args.port)
    driver.attach_sink(sink)

    with driver:
        deadline = time.time() + args.seconds
        try:
            while time.time() < deadline:
                try:
                    frame = sink.get(timeout=0.5)
                    print(f"timestamp: {frame['timestamp']:.3f}")
                    print(f"skew: {frame['skew'] * 1000:.3f} ms")
                    print(f"action_list: {frame['action_list']['axes']}, {frame['action_list']['buttons']}")
                    print(f"left_arm: {frame['joint_states']['left_arm']['position']}")
                    print(f"right_arm: {frame['joint_states']['right_arm']['position']}")
                except queue.Empty:
                    print("waiting for frame...")
        except KeyboardInterrupt:
            # exiting driver context will stop the stream and driver automaitically
            print("Interrupted by user, stopping stream.")


if __name__ == "__main__":
    main()
