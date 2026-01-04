"""
Train ResNet50 transfer learning model.

This script mirrors the workflow in
notebooks/04_transfer_learning_resnet50.ipynb.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

from src.data_loader import collect_image_paths, stratified_split


DATASET_PATH = "data/raw/PlantVillage"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


def main():
    image_paths, labels = collect_image_paths(DATASET_PATH)
    X_train, y_train, X_val, y_val, _, _ = stratified_split(image_paths, labels)

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(set(labels)), activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Training ResNet50 transfer learning model...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    os.makedirs("results/metrics", exist_ok=True)
    model.save("results/metrics/resnet50_tl_model.h5")
    print("ResNet50 model saved.")


if __name__ == "__main__":
    main()
