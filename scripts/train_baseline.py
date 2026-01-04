"""
Train baseline CNN model.

This script reproduces the baseline CNN training performed in
notebooks/02_baseline_cnn.ipynb.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
from src.data_loader import collect_image_paths, stratified_split
from src.models import build_baseline_cnn


DATASET_PATH = "data/raw/PlantVillage"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


def main():
    image_paths, labels = collect_image_paths(DATASET_PATH)
    X_train, y_train, X_val, y_val, _, _ = stratified_split(image_paths, labels)

    model = build_baseline_cnn(
        input_shape=(224, 224, 3),
        num_classes=len(set(labels))
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Training baseline CNN...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    os.makedirs("results/metrics", exist_ok=True)
    model.save("results/metrics/baseline_cnn_model.h5")
    print("Baseline model saved.")


if __name__ == "__main__":
    main()
