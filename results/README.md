# Results Directory

This directory contains **frozen outputs** generated during model training
and analysis. Files here are treated as final artifacts and are not meant
to be regenerated automatically on every run.

## Contents

### metrics/
- `baseline_cnn_model.h5` — Trained baseline CNN model
- `resnet50_tl_model.h5` — Trained ResNet50 transfer learning model
- `baseline_metrics.txt` — Evaluation metrics for baseline model
- `class_names.json` — Class label mapping
- `test_samples.json` — Fixed test samples used for Grad-CAM analysis

### plots/
- Training and validation accuracy/loss curves
- Dataset distribution visualizations

### gradcam/
- Grad-CAM visualizations for baseline CNN

### gradcam_resnet/
- Grad-CAM visualizations for ResNet50 model