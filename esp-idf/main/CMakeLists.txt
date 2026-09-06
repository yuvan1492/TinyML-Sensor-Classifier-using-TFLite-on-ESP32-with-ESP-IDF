idf_component_register(
    SRCS "main.cc"
    INCLUDE_DIRS "."
    REQUIRES esp-tflite-micro esp_driver_uart
    PRIV_REQUIRES esp_timer
)
