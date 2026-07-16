"""Writer process exposing QueueSink over a multiprocessing manager."""

import argparse
import time

from remote_control_lite import CombinedStreamDriver, QueueSinkServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/galbotV1RemoteOperate", help="serial port path")
    parser.add_argument("--host", default="127.0.0.1", help="manager host")
    parser.add_argument("--manager-port", type=int, default=50000, help="manager port")
    parser.add_argument("--authkey", default="queue", help="manager auth key")
    parser.add_argument("--maxsize", type=int, default=128, help="internal queue size")
    parser.add_argument("--seconds", type=float, default=5.0, help="auto stop after N seconds (0=inf)")
    args = parser.parse_args()

    server = QueueSinkServer(
        host=args.host,
        port=args.manager_port,
        authkey=args.authkey,
        maxsize=args.maxsize,
    )

    with server:
        active_host, active_port = server.address
        print(f"Queue manager started at {active_host}:{active_port} (auth={args.authkey})")

        driver = CombinedStreamDriver(port=args.port)
        driver.attach_sink(server.sink)

        deadline = time.time() + args.seconds if args.seconds > 0 else None
        try:
            with driver:
                print("Driver running. Launch reader with queue_sink/reader.py to consume frames.")
                while True:
                    if deadline and time.time() > deadline:
                        break
                    time.sleep(0.5)
        except KeyboardInterrupt:
            print("Interrupted by user, stopping writer.")


if __name__ == "__main__":
    main()
