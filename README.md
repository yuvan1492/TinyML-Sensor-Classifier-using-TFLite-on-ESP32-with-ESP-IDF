# TinyML Sensor Classifier on ESP32 with ESP-IDF

This project trains a small sensor-sequence classifier, converts it to a
fully quantized INT8 TensorFlow Lite model, embeds that model in ESP-IDF
firmware, and runs inference on a real ESP32-WROOM over UART.

There is no Arduino sketch in this project. The firmware is built and flashed
with the Espressif ESP-IDF CLI.

## What This Project Demonstrates

- A complete training-to-device TinyML pipeline.
- Quantization-aware input and output contracts.
- C++ TensorFlow Lite Micro inference on ESP32.
- ESP-IDF component dependencies and managed components.
- Real hardware classification and latency measurement.

## System Flow

```text
Synthetic sensor data
  -> Keras Conv1D classifier
  -> float32 TFLite model
  -> representative-dataset INT8 quantization
  -> sensor_model_int8.tflite
  -> export_header.py
  -> esp-idf/main/model_data.h
  -> ESP-IDF C++ application
  -> idf.py build
  -> idf.py flash
  -> UART line with 32 float samples
  -> int8 quantization on device
  -> TensorFlow Lite Micro Invoke()
  -> NORMAL or ANOMALY with latency
```

## Repository Layout

```text
04-tinyml-esp32/
  train_and_convert.py       Generate data, train, convert, and write metadata
  export_header.py           Convert the INT8 flatbuffer to a C header
  requirements.txt           TensorFlow and NumPy requirements
  artifacts/
    sensor_model.keras       Keras checkpoint
    sensor_model_float32.tflite
    sensor_model_int8.tflite
    metadata.json             Tensor contract and quantization values
  esp-idf/
    CMakeLists.txt            ESP-IDF project
    sdkconfig.defaults        Default ESP-IDF settings
    main/
      CMakeLists.txt          Component dependencies
      idf_component.yml       esp-tflite-micro dependency
      main.cc                 UART parser and on-device inference
      model_data.h            Embedded INT8 model bytes
```

## Model and Data

Each sample is a 32-value window from a sine wave:

- `normal`: sine wave plus small random noise;
- `anomaly`: the same signal with a four-sample spike in the middle.

The Keras model is:

```text
Input(32, 1)
  -> Conv1D(8, kernel_size=3, ReLU)
  -> GlobalAveragePooling1D
  -> Dense(8, ReLU)
  -> Dense(2, softmax)
```

The INT8 model contract is stored in [artifacts/metadata.json](artifacts/metadata.json):

| Property | Value |
| --- | --- |
| Input shape | `[1, 32, 1]` |
| Input type | `int8` |
| Input scale | `0.012561725452542305` |
| Input zero point | `-33` |
| Output shape | `[1, 2]` |
| Output type | `int8` |
| Output scale | `0.00390625` |
| Output zero point | `-128` |

For a float sample `x`, firmware quantizes with:

```text
q = round(x / input_scale) + input_zero_point
```

The output is converted back to a probability-like value with:

```text
value = (q - output_zero_point) * output_scale
```

The ESP32 must use the values from `metadata.json`; it must not assume that
the INT8 tensor accepts raw float values.

## PC Setup

TensorFlow is installed in the project-specific Python 3.11 environment:

```powershell
$py = "C:\Users\41216\Videos\AI_\.venv-tinyml\Scripts\python.exe"
Set-Location "C:\Users\41216\Videos\AI_\04-tinyml-esp32"
& $py -m pip install -r requirements.txt
& $py train_and_convert.py
& $py export_header.py
```

The conversion step produces:

- `artifacts/sensor_model.keras`
- `artifacts/sensor_model_float32.tflite`
- `artifacts/sensor_model_int8.tflite`
- `artifacts/metadata.json`
- `esp-idf/main/model_data.h`

## ESP-IDF Environment

ESP-IDF is installed at:

```text
C:\Users\41216\Videos\esp-idf
```

The Espressif-managed CLI environment is named:

```text
idf6.2_py3.14_env
```

Activate it in a new PowerShell terminal. The process-scoped execution-policy
setting is needed on this Windows setup:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
& "C:\Users\41216\Videos\esp-idf\export.ps1"
idf.py --version
```

## Build and Flash

The tested board is an ESP32-D0WD-V3 connected through a Silicon Labs CP210x
USB-UART bridge on `COM24`.

```powershell
Set-Location "C:\Users\41216\Videos\AI_\04-tinyml-esp32\esp-idf"
idf.py set-target esp32
idf.py -p COM24 build flash
```

The project uses these ESP-IDF dependencies:

- `espressif/esp-tflite-micro`;
- `esp_driver_uart` for UART input;
- `esp_timer` for inference timing.

The firmware registers the operators required by this converted model:

```text
CONV_2D, MEAN, FULLY_CONNECTED, RELU, SOFTMAX, RESHAPE, EXPAND_DIMS
```

`EXPAND_DIMS` is important because TensorFlow Lite lowers the Keras `Conv1D`
path through that operation. Without registering it, `AllocateTensors()` fails
on the ESP32.

## UART Protocol

The firmware listens on the console UART at 115200 baud. After boot it prints:

```text
ready
```

Send exactly 32 comma-separated floating-point values followed by a newline.
The firmware quantizes the line, invokes TFLite Micro, compares the two
dequantized outputs, and prints the predicted class and execution time.

## Verified Results

Training and conversion produced:

| Measurement | Result |
| --- | --- |
| Test accuracy | `1.0` |
| Float32 model | `4028 bytes` |
| INT8 model | `4560 bytes` |
| Application binary | `0x3ae60 bytes`; 77% of the 1 MiB app partition remains free |
| Target | ESP32-D0WD-V3, revision 3.1 |
| Flash port | COM24 |

Real device test through the ESP-IDF monitor and UART:

```text
NORMAL normal=0.953 anomaly=0.047 latency_us=1957
ANOMALY normal=0.051 anomaly=0.949 latency_us=1254
```

The device correctly distinguished both known windows. The measured inference
latency was 1.957 ms for the normal sample and 1.254 ms for the anomaly sample.

## Troubleshooting Notes

### `driver/uart.h: No such file or directory`

Recent ESP-IDF versions place UART in `esp_driver_uart`. Ensure
`main/CMakeLists.txt` contains:

```cmake
REQUIRES esp-tflite-micro esp_driver_uart
PRIV_REQUIRES esp_timer
```

### `EXPAND_DIMS` is not registered

Register the operation and increase the resolver capacity in `main.cc`:

```cpp
static tflite::MicroMutableOpResolver<7> resolver;
resolver.AddExpandDims();
```

### COM port is busy

Close any existing `idf.py monitor` session before flashing. The CP210x board
was detected as `COM24` on the tested machine; your port may differ.

## Next Experiments

1. Feed the classifier from a real sensor rather than UART-generated data.
2. Measure exact tensor arena usage on the final firmware build.
3. Repeat the deployment on ESP32-S3.
4. Evaluate a much smaller character predictor using the measured memory and
   latency budget instead of assuming that the original TinyGPT will fit.
