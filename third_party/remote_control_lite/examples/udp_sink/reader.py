"""Reader that listens for UDP combined frames."""

import argparse
import json
import socket
import time


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
    parser.add_argument("--host", default="0.0.0.0", help="bind host")
    parser.add_argument("--port", type=int, default=12000, help="bind port")
    parser.add_argument("--seconds", type=float, default=30.0, help="run duration")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(1.0)

    deadline = time.time() + args.seconds
    try:
        while time.time() < deadline:
            try:
                data, addr = sock.recvfrom(65535)
                frame = json.loads(data.decode("utf-8"))
                print(addr, _summarize(frame))
            except socket.timeout:
                print("waiting for datagram...")
    except KeyboardInterrupt:
        print("Interrupted by user, stopping reader.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
