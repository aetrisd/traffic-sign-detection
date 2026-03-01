# Traffic Sign & Light Detection System

A computer vision system that detects and classifies traffic signs and traffic lights in real time using a YOLOv8 neural network trained on 6,000+ labeled images across 15 classes.

## Tech Stack

- Python 3.12
- YOLOv8 (Ultralytics) — object detection
- PyTorch with CUDA — GPU-accelerated training
- OpenCV — image processing and visualization
- Pandas / NumPy — data analysis
- Matplotlib — evaluation plots
- Google Colab — training environment

## Features

- Trains a YOLOv8 nano model from scratch on the CarDetection dataset
- Detects 15 classes: traffic lights (red/green), stop signs, and speed limits (10–120 km/h)
- Generates precision-confidence curves and normalized confusion matrices for evaluation
- Interactive inference pipeline — upload an image and receive bounding box predictions with confidence scores
- Dataset class distribution analysis to identify imbalance

## Getting Started

**Open in Google Colab:** [Traffic Sign & Light Detection System](https://colab.research.google.com/drive/1Q4r5p3gJVKWIrehS8O-TM784z8f51yB9)

The notebook is designed to run in Google Colab with GPU acceleration. The interactive inference stage (image upload and real-time prediction) requires the Colab environment and will not work in a standard Jupyter setup.

Run cells sequentially:

1. Install dependencies and download the dataset via Kaggle Hub
2. Analyze class distribution
3. Train the YOLOv8 model (50 epochs, 416×416 resolution — ~30–40 min on GPU)
4. Evaluate with precision-confidence curves and confusion matrix
5. Upload an image and run inference (Colab only)

## Results

- Trained on ~5,600 annotated images
- Model evaluation includes mAP50 score, per-class precision, and inference speed
- Confusion matrix highlights common misclassification patterns between similar speed limit signs

## Project Structure

```
capstone.ipynb       # Main notebook — training, evaluation, inference
```
