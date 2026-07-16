"""Writer that broadcasts combined frames over UDP."""

import argparse
import signal
import time

from remote_control_lite import CombinedStreamDriver, UDPSink


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-port", default="/dev/galbotV1RemoteOperate", help="serial port path")
    parser.add_argument("--host", default="127.0.0.1", help="destination host")
    parser.add_argument("--port", type=int, default=12000, help="destination port")
    parser.add_argument("--seconds", type=float, default=0.0, help="auto stop after N seconds (0=inf)")
    args = parser.parse_args()

    sink = UDPSink(args.host, args.port)
    driver = CombinedStreamDriver(port=args.device_port)
    driver.attach_sink(sink)

    stop = False

    def _handle_stop(*_):
        nonlocal stop
        stop = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_stop)

    deadline = time.time() + args.seconds if args.seconds > 0 else None
    try:
        with driver:
            print(f"Streaming combined frames to udp://{args.host}:{args.port}")
            while not stop:
                if deadline and time.time() > deadline:
                    break
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down writer.")
        stop = True


if __name__ == "__main__":
    main()
