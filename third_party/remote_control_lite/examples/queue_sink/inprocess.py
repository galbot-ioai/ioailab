"""In-process consumer for QueueSink with combined frames."""

import argparse
import queue
import time

from remote_control_lite import CombinedStreamDriver, QueueSink


def _summarize(frame):
    joint = frame.get("joint", {})
    left = joint.get("left", {})
    right = joint.get("right", {})
    skew = frame.get("skew", 0)
    return {
        "timestamp": frame.get("timestamp"),
        "left_seq": left.get("sequence"),
        "right_seq": right.get("sequence"),
        "skew_seq": skew,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/galbotV1RemoteOperate", help="serial port path")
    parser.add_argument("--seconds", type=float, default=30.0, help="run duration")
    args = parser.parse_args()

    sink = QueueSink(maxsize=64)
    driver = CombinedStreamDriver(port=args.port)
    driver.attach_sink(sink)

    with driver:
        deadline = time.time() + args.seconds
        while time.time() < deadline:
            try:
                frame = sink.get(timeout=0.5)
                print(_summarize(frame))
            except queue.Empty:
                print("no frame yet")


if __name__ == "__main__":
    main()
