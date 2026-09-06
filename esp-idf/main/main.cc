// TinyML sensor classifier on ESP32 using esp-tflite-micro over UART.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

// Must match metadata.json produced by train_and_convert.py.
static constexpr int kWindowLength = 32;
static constexpr float kInputScale = 0.012561725452542305f;
static constexpr int kInputZeroPoint = -33;
static constexpr float kOutputScale = 0.00390625f;
static constexpr int kOutputZeroPoint = -128;

static constexpr size_t kTensorArenaSize = 8 * 1024;
alignas(16) static uint8_t g_tensor_arena[kTensorArenaSize];

static const char* TAG = "tinyml";

extern "C" void app_main(void) {
    const tflite::Model* model = tflite::GetModel(g_sensor_model_int8);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "schema mismatch: %lu vs %d",
                 (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    static tflite::MicroMutableOpResolver<7> resolver;
    resolver.AddConv2D();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddExpandDims();

    static tflite::MicroInterpreter interpreter(
        model, resolver, g_tensor_arena, kTensorArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed");
        return;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    ESP_LOGI(TAG, "arena_used=%u input_bytes=%u output_bytes=%u",
             (unsigned)interpreter.arena_used_bytes(),
             (unsigned)input->bytes, (unsigned)output->bytes);

    const uart_port_t uart_num = UART_NUM_0;
    uart_config_t cfg = {};
    cfg.baud_rate = 115200;
    cfg.data_bits = UART_DATA_8_BITS;
    cfg.parity = UART_PARITY_DISABLE;
    cfg.stop_bits = UART_STOP_BITS_1;
    cfg.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
    cfg.source_clk = UART_SCLK_DEFAULT;
    uart_driver_install(uart_num, 1024, 0, 0, nullptr, 0);
    uart_param_config(uart_num, &cfg);

    char line[1024];
    size_t idx = 0;
    printf("ready\n");

    while (true) {
        uint8_t ch;
        int n = uart_read_bytes(uart_num, &ch, 1, portMAX_DELAY);
        if (n <= 0) continue;
        if (ch == '\r') continue;
        if (ch != '\n' && idx + 1 < sizeof(line)) {
            line[idx++] = (char)ch;
            continue;
        }
        line[idx] = '\0';
        idx = 0;

        float samples[kWindowLength];
        int count = 0;
        char* save = nullptr;
        for (char* tok = strtok_r(line, ",", &save);
             tok && count < kWindowLength;
             tok = strtok_r(nullptr, ",", &save)) {
            samples[count++] = strtof(tok, nullptr);
        }
        if (count != kWindowLength) {
            printf("err need %d floats got %d\n", kWindowLength, count);
            continue;
        }

        int8_t* in = input->data.int8;
        for (int i = 0; i < kWindowLength; ++i) {
            int q = (int)lroundf(samples[i] / kInputScale) + kInputZeroPoint;
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            in[i] = (int8_t)q;
        }

        int64_t t0 = esp_timer_get_time();
        if (interpreter.Invoke() != kTfLiteOk) {
            printf("err invoke failed\n");
            continue;
        }
        int64_t t1 = esp_timer_get_time();

        int8_t* out = output->data.int8;
        float p0 = (out[0] - kOutputZeroPoint) * kOutputScale;
        float p1 = (out[1] - kOutputZeroPoint) * kOutputScale;
        const char* label = (p1 > p0) ? "ANOMALY" : "NORMAL";
        printf("%s normal=%.3f anomaly=%.3f latency_us=%lld\n",
               label, p0, p1, (long long)(t1 - t0));
    }
}
