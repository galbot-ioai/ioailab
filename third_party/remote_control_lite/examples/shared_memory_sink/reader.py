"""Reader attaching to existing shared memory sink."""

import argparse
import json
import time

from remote_control_lite import SharedMemorySink


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
    parser.add_argument("--name", required=True, help="shared memory name from writer")
    parser.add_argument("--size", type=int, default=65536, help="shared memory size (match writer)")
    parser.add_argument("--seconds", type=float, default=30.0, help="run duration")
    args = parser.parse_args()
    
    sink = SharedMemorySink(name=args.name, size=args.size)
    sink.open(driver=None)
    try:
        deadline = time.time() + args.seconds
        while time.time() < deadline:
            frame = sink.read()
            if frame:
                print(json.dumps(_summarize(frame), ensure_ascii=False))
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("Interrupted by user, stopping reader.")
    finally:
        sink.close()


if __name__ == "__main__":
    main()
