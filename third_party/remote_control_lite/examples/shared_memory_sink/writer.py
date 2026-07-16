"""Writer that publishes combined frames to shared memory."""

import argparse
import signal
import time

from remote_control_lite import CombinedStreamDriver, SharedMemorySink


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/galbotV1RemoteOperate", help="serial port path")
    parser.add_argument("--size", type=int, default=65536, help="shared memory size")
    parser.add_argument("--seconds", type=float, default=0.0, help="auto stop after N seconds (0=inf)")
    args = parser.parse_args()

    sink = SharedMemorySink(size=args.size)
    driver = CombinedStreamDriver(port=args.port)
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
            print(f"shared memory name: {sink.name}")
            print("Start reader with shared_memory_sink/reader.py --name", sink.name)
            while not stop:
                if deadline and time.time() > deadline:
                    break
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down writer.")
        stop = True


if __name__ == "__main__":
    main()
