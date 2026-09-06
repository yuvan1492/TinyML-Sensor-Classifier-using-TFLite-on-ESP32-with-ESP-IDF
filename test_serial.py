"""Send sensor windows to the ESP32 over UART and print its classification."""
import sys
import time

import serial

PORT = "COM24"
BAUD = 115200

NORMAL = "0.0000,0.2013,0.3944,0.5713,0.7248,0.8486,0.9378,0.9885,0.9987,0.9681,0.8978,0.7908,0.6514,0.4853,0.2994,0.1012,-0.1012,-0.2994,-0.4853,-0.6514,-0.7908,-0.8978,-0.9681,-0.9987,-0.9885,-0.9378,-0.8486,-0.7248,-0.5713,-0.3944,-0.2013,0.0000"
ANOMALY = "0.0000,0.2013,0.3944,0.5713,0.7248,0.8486,0.9378,0.9885,0.9987,0.9681,0.8978,0.7908,0.6514,0.4853,1.7994,1.6012,1.3988,1.2006,-0.4853,-0.6514,-0.7908,-0.8978,-0.9681,-0.9987,-0.9885,-0.9378,-0.8486,-0.7248,-0.5713,-0.3944,-0.2013,0.0000"


def read_available(ser: serial.Serial, seconds: float) -> str:
    end = time.time() + seconds
    buf = b""
    while time.time() < end:
        n = ser.in_waiting
        if n:
            buf += ser.read(n)
        else:
            time.sleep(0.05)
    return buf.decode(errors="replace")


def main():
    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        print(f"--- connected to {PORT} ---")
        print(read_available(ser, 2.0))

        for name, line in (("NORMAL sample", NORMAL), ("ANOMALY sample", ANOMALY)):
            print(f">>> sending {name}")
            ser.write((line + "\n").encode())
            print(read_available(ser, 1.0))


if __name__ == "__main__":
    sys.exit(main())
