"""Consume combined frames via SharedMemorySink in-process."""

import argparse
import json
import time

from remote_control_lite import CombinedStreamDriver, SharedMemorySink


def _summarize(frame):
    joint = frame.get("joint", {})
    left = joint.get("left", {})
    right = joint.get("right", {})
    return {
        "timestamp": frame.get("timestamp"),
        "left_seq": left.get("sequence"),
        "right_seq": right.get("sequence"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/galbotV1RemoteOperate", help="serial port path")
    parser.add_argument("--seconds", type=float, default=30.0, help="run duration")
    parser.add_argument("--size", type=int, default=65536, help="shared memory size")
    args = parser.parse_args()

    sink = SharedMemorySink(size=args.size)
    driver = CombinedStreamDriver(port=args.port)
    driver.attach_sink(sink)

    with driver:
        print(f"shared memory name: {sink.name}")
        deadline = time.time() + args.seconds
        try:
            while time.time() < deadline:
                frame = sink.read()
                if frame:
                    print(json.dumps(_summarize(frame), ensure_ascii=False))
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Interrupted by user, stopping stream.")


if __name__ == "__main__":
    main()
