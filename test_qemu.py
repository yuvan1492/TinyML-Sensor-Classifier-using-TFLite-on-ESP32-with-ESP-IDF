"""Boot the ESP32 firmware in QEMU and verify its UART classifications."""
import argparse
import math
import os
import re
import selectors
import subprocess
import sys
import time
from pathlib import Path


WINDOW_LENGTH = 32
NORMAL = [math.sin(2 * math.pi * index / (WINDOW_LENGTH - 1)) for index in range(WINDOW_LENGTH)]
ANOMALY = [value + 1.5 if 14 <= index < 18 else value for index, value in enumerate(NORMAL)]


def encode_window(values):
    return ",".join(f"{value:.4f}" for value in values)


def read_until(process, selector, output, pattern, timeout):
    deadline = time.monotonic() + timeout
    while not pattern.search(output):
        if process.poll() is not None:
            raise RuntimeError(f"QEMU exited with code {process.returncode}:\n{output.decode(errors='replace')}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for {pattern.pattern}:\n{output.decode(errors='replace')}")
        for key, _ in selector.select(min(remaining, 0.2)):
            chunk = os.read(key.fd, 4096)
            if chunk:
                output.extend(chunk)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qemu", default="qemu-system-xtensa")
    parser.add_argument("--flash-image", type=Path, default=Path("esp-idf/build/merged-binary.bin"))
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    command = [
        args.qemu,
        "-display", "none",
        "-monitor", "none",
        "-serial", "stdio",
        "-machine", "esp32",
        "-drive", f"file={args.flash_image.resolve()},if=mtd,format=raw",
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = bytearray()
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            read_until(process, selector, output, re.compile(rb"\bready\b"), args.timeout)

            for sample, expected in (
                (NORMAL, rb"NORMAL normal=.*latency_us=\d+"),
                (ANOMALY, rb"ANOMALY normal=.*latency_us=\d+"),
            ):
                process.stdin.write((encode_window(sample) + "\n").encode())
                process.stdin.flush()
                read_until(process, selector, output, re.compile(expected), args.timeout)

        print(output.decode(errors="replace"), end="")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, TimeoutError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)