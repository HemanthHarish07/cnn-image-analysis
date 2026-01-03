import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf
import cv2

from src.data_loader import collect_image_paths, stratified_split
from src.models import build_baseline_cnn_functional


# ============================================================
# CONFIG
# ============================================================
DATASET_PATH = "data/raw/PlantVillage"
MODEL_PATH = "results/metrics/baseline_cnn_model.h5"

IMAGE_SIZE = (224, 224)
NUM_SAMPLES = 30

OUTPUT_DIR = "results/gradcam"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "correct"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "incorrect"), exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================
image_paths, labels = collect_image_paths(DATASET_PATH)
_, _, _, _, X_test, y_test = stratified_split(image_paths, labels)

class_names = sorted(set(labels))
class_to_index = {name: idx for idx, name in enumerate(class_names)}
index_to_class = {idx: name for name, idx in class_to_index.items()}

y_test_encoded = np.array([class_to_index[y] for y in y_test])

# Subset (important for stability)
X_test = X_test[:NUM_SAMPLES]
y_test_encoded = y_test_encoded[:NUM_SAMPLES]


# ============================================================
# LOAD MODEL (CRITICAL FIX)
# ============================================================
# Load original Sequential model
model_seq = tf.keras.models.load_model(MODEL_PATH)

# Rebuild same model as Functional
model = build_baseline_cnn_functional(
    input_shape=(224, 224, 3),
    num_classes=len(class_names)
)

# Transfer weights
model.set_weights(model_seq.get_weights())

# Warm-up forward pass (graph build)
_ = model(tf.zeros((1, 224, 224, 3)))


# ============================================================
# GRAD-CAM SETUP
# ============================================================
LAST_CONV_LAYER = "last_conv"

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        model.get_layer(LAST_CONV_LAYER).output,
        model.output
    ]
)


def compute_gradcam(img_tensor, class_idx):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


# ============================================================
# RUN GRAD-CAM ANALYSIS
# ============================================================
for i, img_path in enumerate(X_test):
    # Load image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = img / 255.0
    img_tensor = tf.expand_dims(img, axis=0)

    # Predict
    preds = model(img_tensor, training=False)
    pred_class = tf.argmax(preds, axis=1).numpy()[0]
    true_class = y_test_encoded[i]

    # Grad-CAM
    heatmap = compute_gradcam(img_tensor, pred_class)

    # Overlay
    img_np = (img.numpy() * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

    # Save
    status = "correct" if pred_class == true_class else "incorrect"
    filename = f"{i}_{index_to_class[true_class]}_pred_{index_to_class[pred_class]}.jpg"

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, status, filename),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    print(f"[{i+1}/{NUM_SAMPLES}] saved → {status}")

print("Grad-CAM generation complete.")
