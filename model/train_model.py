"""
Train a MobileNetV2 transfer-learning classifier.

Expected input:
  dataset/
    Tomato___Early_blight/
      image1.jpg
    Tomato___healthy/
      image2.jpg
    Potato___Late_blight/
      image3.jpg

Use a dataset whose license permits your intended academic use.
"""
import json
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../dataset", help="Folder with one subfolder per class")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--output", default=".")
    args = parser.parse_args()

    data_dir = Path(args.data).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise SystemExit(f"Dataset folder not found: {data_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=SEED,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="int"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="int"
    )
    class_names = train_ds.class_names
    if len(class_names) < 2:
        raise SystemExit("At least two class folders are required.")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
    ], name="augmentation")

    base = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            str(output_dir / "crop_disease_model.keras"),
            save_best_only=True, monitor="val_accuracy"
        )
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    (output_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print("Saved model and class_names.json. Evaluate on a separate, unseen test set before reporting results.")

if __name__ == "__main__":
    main()
