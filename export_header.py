"""Export the INT8 TFLite flatbuffer as a C header."""
from pathlib import Path

MODEL_PATH = Path("artifacts/sensor_model_int8.tflite")
HEADER_PATH = Path("arduino/TinyMLSensor/model_data.h")
SYMBOL = "g_sensor_model_int8"


def main():
    data = MODEL_PATH.read_bytes()
    HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)
    values = ", ".join(f"0x{byte:02x}" for byte in data)
    header = (
        "#pragma once\n"
        "#include <stdint.h>\n\n"
        f"const unsigned char {SYMBOL}[] = {{\n    {values}\n}};\n"
        f"const unsigned int {SYMBOL}_len = {len(data)};\n"
    )
    HEADER_PATH.write_text(header, encoding="ascii")
    print(f"wrote {HEADER_PATH} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
