"""
predict.py
----------------
Shared image-preprocessing + prediction helper used by app.py,
and also runnable standalone from the command line:

    python predict.py path/to/image.png
"""

import os
import sys
import numpy as np
import cv2
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join("model", "digit_model.h5")
_model = None


def get_model():
    """Lazy-load and cache the trained model."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train_model.py first."
            )
        _model = load_model(MODEL_PATH)
    return _model


def preprocess_image(image_path):
    """
    Preprocess an uploaded image so it matches the MNIST format:
      - Convert to grayscale
      - Resize to 28x28
      - Invert colors if the digit is dark-on-light (MNIST is light-on-dark)
      - Normalize pixel values to 0-1
      - Flatten to a 784-length vector
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Convert RGB -> Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # MNIST digits are white strokes on a black background.
    # If the uploaded image looks like black strokes on a white
    # background (mean pixel value is high/light), invert it.
    if np.mean(resized) > 127:
        resized = cv2.bitwise_not(resized)

    # Normalize
    normalized = resized.astype("float32") / 255.0

    # Flatten
    flattened = normalized.reshape(1, 784)

    return flattened, resized


def predict_digit(image_path):
    """
    Runs the full prediction pipeline on an image path.
    Returns (predicted_digit, confidence_percent, all_probabilities)
    """
    model = get_model()
    flattened, _ = preprocess_image(image_path)

    probabilities = model.predict(flattened, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities) * 100)

    return predicted_digit, confidence, probabilities.tolist()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    digit, confidence, _ = predict_digit(image_path)
    print(f"Predicted Digit : {digit}")
    print(f"Confidence      : {confidence:.2f}%")
