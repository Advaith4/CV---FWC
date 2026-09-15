# Practical Computer Vision Exercises (Module 6)

This repository contains the interactive Jupyter Notebook implementation of all 36 exercises from **Module 6 - Practical Computer Vision Exercises & Evaluation Questions**.

## 📌 Contents

- **`computer_vision_exercises.ipynb`**: Complete, self-contained, executed Jupyter Notebook covering all 36 exercises across 5 phases:
  - **Phase 1 (Ex 1–7):** Spatial Filters & Classical CV (Pixel arrays, Vertical/Horizontal edge detectors, Sobel gradients, Box averaging blur, Gaussian blur + Canny edge detection, Laplacian sharpening).
  - **Phase 2 (Ex 8–10):** Color Spaces & Deterministic Vision (RGB channel inspection, HSV color conversion, Green object segmentation mask).
  - **Phase 3 (Ex 11–25):** Data Augmentations (Geometric transforms, Color jitters, Gaussian blur, Perspective, Random Erasing, Composed pipeline, Augmentation policy matrix).
  - **Phase 4 (Ex 26–33):** Deep Learning & CNN Building Blocks (Learnable `nn.Conv2d`, Padding, Stride, `nn.ReLU`, `nn.MaxPool2d`, Tiny CNN architecture, Logits output).
  - **Phase 5 (Ex 34–36):** Evaluation Metrics & Architectures (Accuracy paradox simulation, Confusion matrix computation, Landmark architectures summary: LeNet, AlexNet, VGG, Inception, ResNet).
- **`image.jpg`**: Sample pet image used across the image processing and augmentation exercises.
- **`Module_6_Practical_Computer_Vision_Exercises.pdf`**: Original curriculum documentation.

## 🚀 Setup & Execution

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# 2. Install dependencies
pip install numpy pillow opencv-python matplotlib scikit-learn torch torchvision jupyter

# 3. Launch the notebook
jupyter notebook computer_vision_exercises.ipynb
```
