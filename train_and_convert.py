"""Train a tiny sensor classifier and create a fully INT8 TFLite model."""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

SEED = 1337
WINDOW_LENGTH = 32
ARTIFACTS = Path("artifacts")


def make_dataset(samples: int = 4000):
    rng = np.random.default_rng(SEED)
    time = np.linspace(0.0, 2.0 * np.pi, WINDOW_LENGTH, dtype=np.float32)
    normal = np.sin(time)[None, :] + rng.normal(0.0, 0.08, (samples // 2, WINDOW_LENGTH))
    anomaly = np.sin(time)[None, :] + rng.normal(0.0, 0.08, (samples // 2, WINDOW_LENGTH))
    anomaly[:, WINDOW_LENGTH // 2 - 2:WINDOW_LENGTH // 2 + 2] += 1.5

    x = np.concatenate([normal, anomaly], axis=0).astype(np.float32)[..., None]
    y = np.concatenate([
        np.zeros(samples // 2, dtype=np.int32),
        np.ones(samples // 2, dtype=np.int32),
    ])
    order = rng.permutation(len(x))
    split = int(len(x) * 0.8)
    x, y = x[order], y[order]
    return (x[:split], y[:split]), (x[split:], y[split:])


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW_LENGTH, 1), name="sensor_window"),
        tf.keras.layers.Conv1D(8, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax", name="class_probability"),
    ])


def representative_dataset(x_train):
    for sample in x_train[:200]:
        yield [sample[None, ...].astype(np.float32)]


def convert_int8(model, x_train):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(x_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def main():
    tf.keras.utils.set_random_seed(SEED)
    ARTIFACTS.mkdir(exist_ok=True)
    (x_train, y_train), (x_test, y_test) = make_dataset()

    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_split=0.2, epochs=8, batch_size=32, verbose=2)
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    model.save(ARTIFACTS / "sensor_model.keras")

    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_model = float_converter.convert()
    (ARTIFACTS / "sensor_model_float32.tflite").write_bytes(float_model)

    int8_model = convert_int8(model, x_train)
    (ARTIFACTS / "sensor_model_int8.tflite").write_bytes(int8_model)

    interpreter = tf.lite.Interpreter(model_content=int8_model)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    metadata = {
        "window_length": WINDOW_LENGTH,
        "class_names": ["normal", "anomaly"],
        "test_accuracy": float(accuracy),
        "input_shape": input_info["shape"].tolist(),
        "input_dtype": str(input_info["dtype"]),
        "input_scale": float(input_info["quantization_parameters"]["scales"][0]),
        "input_zero_point": int(input_info["quantization_parameters"]["zero_points"][0]),
        "output_shape": output_info["shape"].tolist(),
        "output_dtype": str(output_info["dtype"]),
        "output_scale": float(output_info["quantization_parameters"]["scales"][0]),
        "output_zero_point": int(output_info["quantization_parameters"]["zero_points"][0]),
    }
    (ARTIFACTS / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"float32_bytes={len(float_model)} int8_bytes={len(int8_model)}")


if __name__ == "__main__":
    main()
