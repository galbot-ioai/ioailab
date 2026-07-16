"""Reader that connects to QueueSink writer via multiprocessing manager."""

import argparse
import json
import queue
import time

from remote_control_lite import QueueSinkClient


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
    parser.add_argument("--host", default="127.0.0.1", help="manager host")
    parser.add_argument("--manager-port", type=int, default=50000, help="manager port")
    parser.add_argument("--authkey", default="queue", help="manager auth key")
    parser.add_argument("--seconds", type=float, default=30.0, help="run duration")
    args = parser.parse_args()

    client = QueueSinkClient(host=args.host, port=args.manager_port, authkey=args.authkey)

    with client:
        q = client.queue
        deadline = time.time() + args.seconds
        try:
            while time.time() < deadline:
                try:
                    frame = q.get(timeout=1.0)
                    print(json.dumps(_summarize(frame), ensure_ascii=False))
                except queue.Empty:
                    print("waiting for frame...")
        except KeyboardInterrupt:
            print("Interrupted by user, stopping reader.")


if __name__ == "__main__":
    main()
