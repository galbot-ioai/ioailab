"""Basic example streaming a single arm side over an in-process queue."""

import argparse
import queue
import time

from remote_control_lite import QueueSink, SingleStreamDriver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/galbotV1RemoteOperate", help="serial port path")
    parser.add_argument("--seconds", type=float, default=60.0, help="stream duration")
    parser.add_argument("--side", choices=["left", "right"], default="left", help="arm side")
    parser.add_argument("--stream", choices=["joint", "joystick"], default="joint")
    args = parser.parse_args()

    sink = QueueSink(maxsize=32)
    driver = SingleStreamDriver(side=args.side, stream=args.stream, port=args.port)
    driver.attach_sink(sink)

    with driver:
        deadline = time.time() + args.seconds
        try:
            while time.time() < deadline:
                try:
                    sample = sink.get(timeout=0.5)
                    position = sample["position"]
                    print(
                        f"seq={sample['sequence']} t={sample['timestamp']:.3f} "
                        f"q0={position[0]:.3f}"
                    )
                except queue.Empty:
                    print("no data yet")
        except KeyboardInterrupt:
            # exiting driver context will stop the stream and driver automaitically
            print("Interrupted by user, stopping stream.")


if __name__ == "__main__":
    main()
