# Handwritten Digit Recognition System Using Artificial Neural Networks (ANN)

A Machine Learning web app that recognizes handwritten digits (0–9) from an
uploaded image using an Artificial Neural Network trained on the MNIST dataset.

## Folder Structure

```
Handwritten_Digit_Detection/
│── app.py
│── train_model.py
│── predict.py
│── requirements.txt
│── README.md
│
├── model/
│     digit_model.h5        (created after training)
│
├── static/
│     style.css
│     script.js
│
├── templates/
│     index.html
│
└── uploads/                 (uploaded images are stored here)
```

## Setup (VS Code)

1. Open this folder in VS Code.
2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Step 1 — Train the Model

Run this once. It downloads MNIST, trains the ANN (~18 epochs), and saves
`model/digit_model.h5`.

```bash
python train_model.py
```

Expect ~97-98% test accuracy. Training takes a few minutes on CPU.

## Step 2 — Run the Web App

```bash
python app.py
```

Open your browser at:

```
http://127.0.0.1:3000
```

The app runs on **port 3000** as requested.

## How It Works

1. Upload a handwritten digit image (drag-and-drop or click to browse).
2. Click **Predict Digit**.
3. The image is preprocessed (grayscale → 28×28 → normalized → flattened)
   exactly as described in the project's image-preprocessing pipeline.
4. The ANN predicts probabilities for digits 0–9 via Softmax.
5. The predicted digit, confidence percentage, and a probability breakdown
   for all 10 digits are displayed.

## CLI Prediction (optional)

You can also predict from the command line without the web UI:

```bash
python predict.py path/to/your/image.png
```

## Notes

- Best results come from images with a single digit, reasonably centered,
  similar in style to MNIST (bold strokes, clear background).
- The preprocessing step automatically inverts colors if you upload a
  dark-digit-on-light-background image (MNIST format is light-on-dark).
- Max upload size is 5 MB; accepted formats: png, jpg, jpeg, bmp, gif, webp.
