"""
train_model.py
----------------
Trains an Artificial Neural Network (ANN) on the MNIST dataset
to recognize handwritten digits (0-9), then saves the trained
model to model/digit_model.h5

Run this once before starting app.py:
    python train_model.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "digit_model.h5")


def load_data():
    print("Step 1: Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(f"Training images: {x_train.shape[0]}, Testing images: {x_test.shape[0]}")
    return (x_train, y_train), (x_test, y_test)


def preprocess_data(x_train, y_train, x_test, y_test):
    print("Step 2: Normalizing images (0-255 -> 0-1)...")
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    print("Step 3: Flattening images (28x28 -> 784)...")
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def build_model():
    print("Step 4: Building ANN architecture...")
    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    return model


def compile_model(model):
    print("Step 5: Compiling model (Adam optimizer, categorical crossentropy)...")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, x_train, y_train, x_test, y_test):
    print("Step 6: Training model (15-20 epochs, batch size 32)...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=18,
        batch_size=32,
        verbose=1,
    )
    return history


def evaluate_model(model, x_test, y_test):
    print("Step 7: Evaluating model on test set...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return loss, accuracy


def save_model(model):
    print("Step 8: Saving model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)

    model = build_model()
    model = compile_model(model)
    model.summary()

    train_model(model, x_train, y_train, x_test, y_test)
    evaluate_model(model, x_test, y_test)
    save_model(model)

    print("\nTraining complete! You can now run: python app.py")


if __name__ == "__main__":
    main()
