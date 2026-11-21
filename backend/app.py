
import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "crop_disease_model.keras"
CLASSES_PATH = MODEL_DIR / "class_names.json"

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

_model = None
_class_names = None


def load_artifacts():
    global _model, _class_names

    if _model is None:
        if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
            raise FileNotFoundError(
                "Trained model not found. Follow README.md to train the model first."
            )

        _model = load_model(
            MODEL_PATH,
            custom_objects={"preprocess_input": preprocess_input},
            compile=False
        )

        _class_names = json.loads(
            CLASSES_PATH.read_text(encoding="utf-8")
        )

        if len(_class_names) != _model.output_shape[-1]:
            raise ValueError(
                f"Class count mismatch: JSON has {len(_class_names)} classes, "
                f"but model outputs {_model.output_shape[-1]} classes."
            )

    return _model, _class_names


def friendly_label(raw_label):
    # Convert PlantVillage labels into readable crop and disease names.
    parts = raw_label.replace("___", " - ").replace("_", " ").split(" - ")

    crop = parts[0].strip() if parts else "Unknown crop"

    condition = (
        parts[1].replace("_", " ").strip()
        if len(parts) > 1
        else "Unknown condition"
    )

    healthy = "healthy" in condition.lower()

    return {
        "crop": crop,
        "condition": condition,
        "status": "Healthy" if healthy else "Possible disease detected"
    }


@app.get("/api/health")
def health():
    ready = MODEL_PATH.exists() and CLASSES_PATH.exists()

    return jsonify({
        "status": "ok",
        "model_ready": ready,
        "message": (
            "Model is ready."
            if ready
            else "Train the model before requesting predictions."
        )
    })


@app.post("/api/predict")
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "Upload an image using the 'image' field."
        }), 400

    image_file = request.files["image"]

    if not image_file.filename:
        return jsonify({"error": "No file selected."}), 400

    allowed = {".jpg", ".jpeg", ".png", ".webp"}

    extension = Path(image_file.filename).suffix.lower()

    if extension not in allowed:
        return jsonify({
            "error": "Unsupported file. Use JPG, JPEG, PNG, or WEBP."
        }), 400

    try:
        model, class_names = load_artifacts()

        # Read and resize the uploaded image.
        image = (
            Image.open(image_file.stream)
            .convert("RGB")
            .resize((224, 224))
        )

        # Convert image to a NumPy array.
        array = np.asarray(image, dtype=np.float32)

        # Add batch dimension.
        array = np.expand_dims(array, axis=0)

        # IMPORTANT:
        # MobileNetV2 preprocessing is already included
        # inside the trained model. Do not apply it again here.
        probabilities = model.predict(array, verbose=0)[0]

        # Get the highest prediction.
        best_index = int(np.argmax(probabilities))

        confidence = float(probabilities[best_index])

        raw_label = class_names[best_index]

        details = friendly_label(raw_label)

        # Generate advice.
        if confidence < 0.55:
            advice = [
                "The model is uncertain. Capture a clear, well-lit photo of the affected leaf.",
                "Check multiple plants and consult a local agricultural extension officer before treatment."
            ]

        elif details["status"] == "Healthy":
            advice = [
                "Continue regular crop monitoring.",
                "Use clean tools and follow locally recommended preventive practices."
            ]

        else:
            advice = [
                "Inspect nearby plants for similar symptoms and monitor whether the affected area is spreading.",
                "Avoid applying pesticides based only on this AI result.",
                "Confirm the diagnosis with a qualified agricultural expert and follow locally approved guidance."
            ]

        # Get the top three predictions.
        top_indices = np.argsort(probabilities)[::-1][:3]

        top_predictions = [
            {
                "label": class_names[int(i)],
                "confidence": round(
                    float(probabilities[int(i)]) * 100, 2
                )
            }
            for i in top_indices
        ]

        # Save prediction history.
        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": Path(image_file.filename).name,
            "crop": details["crop"],
            "condition": details["condition"],
            "status": details["status"],
            "confidence": round(confidence * 100, 2),
            "advice": advice,
            "top_predictions": top_predictions
        }

        history_path = ROOT / "data" / "prediction_history.jsonl"

        history_path.parent.mkdir(parents=True, exist_ok=True)

        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        return jsonify(record)

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    except Exception as exc:
        return jsonify({
            "error": f"Could not process image: {exc}"
        }), 500


@app.get("/api/history")
def history():

    path = ROOT / "data" / "prediction_history.jsonl"

    if not path.exists():
        return jsonify([])

    records = []

    for line in path.read_text(encoding="utf-8").splitlines()[-50:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return jsonify(list(reversed(records)))


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False
    )
