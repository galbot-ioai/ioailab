"""In-process UDP streaming with local listener."""

import argparse
import json
import queue
import socket
import threading
import time

from remote_control_lite import CombinedStreamDriver, UDPSink


class _UdpListener(threading.Thread):
    def __init__(self, host: str, port: int, out_queue: "queue.Queue[str]"):
        super().__init__(daemon=True)
        self._host = host
        self._port = port
        self._queue = out_queue
        self._stop_e = threading.Event()

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self._host, self._port))
        sock.settimeout(0.5)
        try:
            while not self._stop_e.is_set():
                try:
                    data, _ = sock.recvfrom(65535)
                    self._queue.put_nowait(data.decode("utf-8"))
                except socket.timeout:
                    continue
                except queue.Full:
                    self._queue.get_nowait()
        finally:
            sock.close()

    def stop(self) -> None:
        self._stop_e.set()


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
    parser.add_argument("--device-port", default="/dev/galbotV1RemoteOperate", help="serial port path")
    parser.add_argument("--seconds", type=float, default=30.0, help="run duration")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=12000)
    args = parser.parse_args()

    udp_sink = UDPSink(args.listen_host, args.listen_port)
    driver = CombinedStreamDriver(port=args.device_port)
    driver.attach_sink(udp_sink)

    messages: "queue.Queue[str]" = queue.Queue(maxsize=16)
    listener = _UdpListener(args.listen_host, args.listen_port, messages)
    listener.start()

    try:
        with driver:
            deadline = time.time() + args.seconds
            try:
                while time.time() < deadline:
                    try:
                        payload = messages.get(timeout=1.0)
                        frame = json.loads(payload)
                        print(_summarize(frame))
                    except queue.Empty:
                        print("waiting for datagram...")
            except KeyboardInterrupt:
                print("Interrupted by user, stopping stream.")
    finally:
        listener.stop()
        listener.join()


if __name__ == "__main__":
    main()
