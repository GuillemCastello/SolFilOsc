"""CNN noise-model utilities used by the oscillation detector."""

import os

# Reserve GPU 1 for other workloads by exposing only physical GPU 0 to TensorFlow.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, layers

class CNN(Model):
    def __init__(self):
        super().__init__()
        self.original_dim = 2000
        self.dropout_rate = 0.2

        self.conv_block = tf.keras.Sequential([
            layers.Input(shape=(self.original_dim, 1)),

            layers.Conv1D(64, 5, activation="relu", padding="valid"),
            layers.Conv1D(64, 8, activation="relu", padding="valid"),
            layers.MaxPooling1D(2),

            layers.Conv1D(64, 11, activation="relu", padding="valid"),
            layers.Conv1D(64, 10, activation="relu", padding="valid"),
            layers.MaxPooling1D(2),

            layers.Conv1D(64, 11, activation="relu", padding="valid"),
            layers.Conv1D(64, 10, activation="relu", padding="valid"),
            layers.MaxPooling1D(2),

            layers.Conv1D(64, 11, activation="relu", padding="valid"),
            layers.Conv1D(64, 10, activation="relu", padding="valid"),
            layers.MaxPooling1D(2),

            layers.Flatten(),

            layers.Dense(1024, activation="relu"),
            layers.Dropout(self.dropout_rate),

            layers.Dense(512, activation="relu"),
            layers.Dropout(self.dropout_rate),

            layers.Dense(256, activation="relu"),
            layers.Dropout(self.dropout_rate),

            layers.Dense(3, activation="sigmoid"),
        ])

    def call(self, input_features):
        x = tf.expand_dims(input_features, -1)
        return self.conv_block(x)

def load_cnn(weights_path: str) -> CNN:
    model = CNN()
    model.load_weights(weights_path)
    return model

def _build_scaler() -> MinMaxScaler:
    sc = MinMaxScaler()
    sc.min_ = np.array([1.5661751, 0.19527236, 2.219622], dtype=np.float32)
    sc.scale_ = np.array([0.14172548, 2.0891168, 0.396035], dtype=np.float32)
    return sc

def cnn_predict_noise_params(model: CNN, psds: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    raw = model.predict(psds, verbose=0)
    raw = np.asarray(raw, np.float64)

    if raw.ndim != 2 or raw.shape[1] != 3:
        raise RuntimeError(f"CNN returned unexpected shape: {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise RuntimeError("CNN prediction contains non-finite values.")

    rescaled = scaler.inverse_transform(raw)
    if not np.all(np.isfinite(rescaled)):
        raise RuntimeError("Scaler inverse_transform produced non-finite values.")

    params = (10.0 ** rescaled).astype(np.float64)
    if not np.all(np.isfinite(params)):
        raise RuntimeError("Noise parameters contain non-finite values after exponentiation.")

    return params
