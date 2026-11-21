
"""Evaluate a saved model on an independent validation directory."""
import json
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Validation folder with class subfolders")
    p.add_argument("--model", default="crop_disease_model.keras")
    p.add_argument("--classes", default="class_names.json")
    args = p.parse_args()

    # Load the model with MobileNetV2's preprocessing function registered.
    model = tf.keras.models.load_model(
        args.model,
        custom_objects={"preprocess_input": preprocess_input},
        compile=False
    )

    ds = tf.keras.utils.image_dataset_from_directory(
        args.data,
        image_size=(224, 224),
        batch_size=32,
        shuffle=False
    )

    # Verify that validation class ordering matches training class ordering.
    names = json.loads(
        Path(args.classes).read_text(encoding="utf-8")
    )

    if ds.class_names != names:
        raise SystemExit(
            "ERROR: Validation class names/order do not match class_names.json.\n"
            f"Validation: {ds.class_names}\n"
            f"Training: {names}"
        )

    probs = model.predict(ds)
    y_pred = probs.argmax(axis=1)
    y_true = np.concatenate([y.numpy() for _, y in ds])

    print("\nClassification Report:\n")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=names,
            zero_division=0
        )
    )

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
