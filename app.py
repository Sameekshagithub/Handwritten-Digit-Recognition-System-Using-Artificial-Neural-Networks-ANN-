"""
app.py
----------------
Flask web application for the Handwritten Digit Recognition System.

Run:
    python app.py

Then open:
    http://127.0.0.1:3000
"""

import os
import uuid
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

from predict import predict_digit, get_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "webp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "digitImage" not in request.files:
        return jsonify({"success": False, "error": "No file part in the request."}), 400

    file = request.files["digitImage"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Unsupported file type."}), 400

    # Save with a unique filename to avoid collisions
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    safe_name = secure_filename(unique_name)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    file.save(save_path)

    try:
        digit, confidence, probabilities = predict_digit(save_path)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

    image_url = url_for("static_uploads", filename=safe_name)

    return jsonify(
        {
            "success": True,
            "digit": digit,
            "confidence": round(confidence, 2),
            "probabilities": [round(p * 100, 2) for p in probabilities],
            "image_url": image_url,
        }
    )


@app.route("/uploads/<path:filename>")
def static_uploads(filename):
    """Serve uploaded images back to the browser for preview."""
    from flask import send_from_directory
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.errorhandler(413)
def file_too_large(_e):
    return jsonify({"success": False, "error": "File too large. Max size is 5MB."}), 413


if __name__ == "__main__":
    # Load model once at startup so the first prediction isn't slow
    print("Loading trained ANN model...")
    get_model()
    print("Model loaded. Starting Flask server on port 3000...")
    app.run(host="0.0.0.0", port=3000, debug=True)
