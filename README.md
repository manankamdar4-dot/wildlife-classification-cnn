# Wildlife Camera-Trap Species Classification

An image classification project that automatically identifies wildlife species (buffalo, elephant, rhino, zebra) from camera-trap style photos using deep learning. Built as a demonstration of transfer learning, comparative model evaluation, and explainable AI — relevant to IoT-based smart camera-trap systems used in conservation and anti-poaching efforts.

## Problem Statement

Camera traps deployed in forests and wildlife reserves capture thousands of images, most of which require manual review to identify which species (if any) triggered the camera. This is slow and doesn't scale, especially when fast alerts are needed — for example, flagging an elephant near a village boundary in near real time. These devices are also typically battery-powered and placed in areas with no reliable internet connection, meaning any AI model deployed on or near them must be lightweight enough to run locally.

This project explores whether a small, camera-based AI system could automatically classify wildlife species — a capability directly relevant to smart camera traps and IoT-integrated conservation monitoring.

## Dataset

**African Wildlife Dataset** — a public dataset of ~1,500 images across 4 classes: buffalo, elephant, rhino, and zebra. Images vary in background, lighting, and angle, closer to real field conditions than a plain studio background.

Source: [biancaferreira/african-wildlife](https://www.kaggle.com/datasets/biancaferreira/african-wildlife) (Kaggle)

## Approach

We used **transfer learning** — starting from models pretrained on ImageNet (1M+ general images) and fine-tuning only the final classification layer on our 4 wildlife categories. This lets us train accurate models in minutes instead of needing to train from scratch on a small dataset.

Two architectures were trained and compared:

| Model | Why it was chosen |
|---|---|
| MobileNetV2 | Lightweight, fast inference — suited for edge/IoT deployment on camera-trap hardware |
| ResNet18 | Deeper architecture, generally higher accuracy — used as a comparison baseline |

Data augmentation (random flips, rotations, color jitter) was applied during training to improve generalization to varied field conditions.

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| MobileNetV2 | 97.80% | 97.86% | 97.80% | 97.81% |
| ResNet18 | 96.48% | 96.66% | 96.48% | 96.49% |

**Note:** Recall is especially important in this task — for a rare or endangered species like rhino, missing a genuine sighting (false negative) is a more costly error than a false alarm.

Confusion matrices for both models are included in this repo (`confusion_matrix_MobileNetV2.png`, `confusion_matrix_ResNet18.png`).

## Explainability (Grad-CAM)

To verify the models were learning meaningful visual features (not just background shortcuts, e.g. "grassy background → zebra"), we generated **Grad-CAM heatmaps**, which highlight the image regions each model focused on to make its prediction. Warmer colors (red/yellow) indicate higher attention. In both models, the heatmaps concentrate on the animal's body rather than the surrounding scenery, supporting that the learned features are genuine.

See `gradcam_MobileNetV2.png` and `gradcam_ResNet18.png` for sample visualizations.

## Project Structure

```
├── wildlife_classification_project.ipynb   # Main notebook (run in Google Colab)
├── mobilenetv2_wildlife_classifier.pth     # Trained MobileNetV2 weights
├── resnet18_wildlife_classifier.pth        # Trained ResNet18 weights
├── model_comparison_results.csv            # Accuracy/precision/recall/F1 comparison
├── confusion_matrix_MobileNetV2.png
├── confusion_matrix_ResNet18.png
├── gradcam_MobileNetV2.png
├── gradcam_ResNet18.png
└── README.md
```

## How to Run

1. Open `wildlife_classification_project.ipynb` in [Google Colab](https://colab.research.google.com)
2. Go to Runtime → Change runtime type → select T4 GPU
3. Generate a Kaggle API token (Kaggle → Settings → API Tokens → Generate New Token) and add it as a Colab secret named `KAGGLE_API_TOKEN`
4. Run all cells from top to bottom

## Future Work

- Convert the MobileNetV2 model to **TensorFlow Lite / ONNX** for deployment on edge devices (e.g., Raspberry Pi, ESP32-CAM), enabling real-time on-device species classification without needing an internet connection.
- Expand the dataset with more species and add a "no animal detected" background class, closer to real camera-trap conditions.
- Test robustness under night-vision/infrared imagery, which many real camera traps rely on.
- Integrate with an actual IoT camera-trap setup for live field deployment and alerting.

## Tech Stack

Python, PyTorch, torchvision, scikit-learn, Grad-CAM, Google Colab (T4 GPU)
