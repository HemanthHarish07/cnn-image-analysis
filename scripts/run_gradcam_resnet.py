import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf
import cv2

from tensorflow.keras.applications.resnet50 import preprocess_input


# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "results/metrics/resnet50_tl_model.h5"
IMAGE_SIZE = (224, 224)
NUM_SAMPLES = 30

OUTPUT_DIR = "results/gradcam_resnet"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "correct"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "incorrect"), exist_ok=True)


# ============================================================
# LOAD TEST SAMPLES (FROZEN SPLIT)
# ============================================================
with open("results/metrics/test_samples.json", "r") as f:
    test_data = json.load(f)

X_test = test_data["paths"]
y_test_encoded = np.array(test_data["labels"])

with open("results/metrics/class_names.json", "r") as f:
    class_names = json.load(f)

index_to_class = {idx: name for idx, name in enumerate(class_names)}

# Subset for stability
X_test = X_test[:NUM_SAMPLES]
y_test_encoded = y_test_encoded[:NUM_SAMPLES]


# ============================================================
# LOAD RESNET MODEL
# ============================================================
model = tf.keras.models.load_model(MODEL_PATH)

# Build graph
_ = model(tf.zeros((1, 224, 224, 3)))


# ============================================================
# GRAD-CAM MODEL
# ============================================================
LAST_CONV_LAYER = "conv5_block3_out"

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

        # Safety unwrap
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


# ============================================================
# RUN GRAD-CAM
# ============================================================
for i, img_path in enumerate(X_test):
    # Load image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)

    #CRITICAL FIX — SAME PREPROCESSING AS TRAINING
    img = preprocess_input(img)

    img_tensor = tf.expand_dims(img, axis=0)

    # Predict
    preds = model(img_tensor, training=False)
    pred_class = tf.argmax(preds, axis=1).numpy()[0]
    true_class = y_test_encoded[i]

    # Grad-CAM
    heatmap = compute_gradcam(img_tensor, pred_class)

    # Visualization (convert back to uint8 for display)
    img_vis = tf.image.resize(img_tensor[0], IMAGE_SIZE)
    img_vis = (img_vis - tf.reduce_min(img_vis)) / (
        tf.reduce_max(img_vis) - tf.reduce_min(img_vis) + 1e-8
    )
    img_vis = (img_vis.numpy() * 255).astype(np.uint8)

    heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(img_vis, 0.6, heatmap_colored, 0.4, 0)

    status = "correct" if pred_class == true_class else "incorrect"
    filename = f"{i}_{index_to_class[true_class]}_pred_{index_to_class[pred_class]}.jpg"

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, status, filename),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    print(f"[{i+1}/{NUM_SAMPLES}] saved → {status}")

print("✅ ResNet Grad-CAM generation complete.")
