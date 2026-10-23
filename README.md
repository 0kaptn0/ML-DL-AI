# 1: Two-Stage Traffic Sign Detection and Classification Pipeline

This repository implements a production-grade, two-stage computer vision cascade for real-time Traffic Sign Detection and Classification, combining state-of-the-art Deep Learning models optimized for deployment on edge devices and embedded systems.

## 🚀 System Architecture Overview

Most monolithic object detectors struggle with fine-grained multi-class classification on small objects like traffic signs. To solve this, we implement a **two-stage cascade pipeline**:

## Stage 1: Single-Class YOLO Detector
Inspired by real-time object detection paradigms (Redmon et al., 2016), a lightweight YOLO model
was configured:

• Architecture: 130 layers, 3,011,043 parameters (size 11.5 MB ,FP32 memory footprint), 8.2
GFLOPs.

• Training Parameters: Input size 1024 x 1024, batch size 32, trained over 100 epochs using
the SGD optimizer with momentum (0.937) and initial learning rate 0.001.

## Stage 2 Model A: Custom CNN Baseline
Designed from scratch using Keras/TensorFlow to achieve a low parameter count:

• Structure: 3 Sequential Convolutional blocks (Conv2D ---> BatchNormalization ---> ReLU --->
MaxPooling2D ---> Dropout. Filters scale from 32 to 64 to 128 (3x3 kernels). Dense
classification head: 128 units, Dropout (0.5), Softmax output across 43 classes.

• Parameter Footprint: 164,459 trainable parameters ( 642.4 KB, FP32 footprint).

• Training Parameters: Trained using AdamW optimizer, sparse categorical cross-entropy loss,
batch size 64, for 50 epochs.

## Stage 2 Model B: MobileNetV3 Small Transfer Learning
• Structure: Torchvision implementation of MobileNetV3 Small initialized with ImageNet pretrained weights. 
  The final linear classifier layer was reconfigured for 43 output classes (in_features=1024, out_features=43). Depthwise separable convolutions significantly reduce standard matrix multiplication FLOPs (Lee & Kim, 2018).
  
• Parameter Footprint: 1,523,771 parameters (6.1 MB, FP32 footprint).

• Training Parameters: Fine-tuned using AdamW (learning rate 0.0001), Cross-Entropy Loss,
batch size 64, for 20 epochs in PyTorch.

```

  [Input Frame] 
        │
        ▼
┌─────────────────┐
│ Stage 1: YOLO   │ ──► Locates sign & extracts Bounding Box
└─────────────────┘
        │ (Crop & Transform)
        ▼
┌─────────────────┐
│ Stage 2:        │ ──► Classifies fine-grained type (43 classes)
│ CNN /           │ 
│ MobileNetV3     │    
└─────────────────┘
        │
        ▼
  [Annotated Output]
```

---

## 📊 Performance & Optimization Benchmarks

To emulate edge deployment (e.g., Raspberry Pi, NVIDIA Jetson, or Mobile Devices), we applied advanced acceleration and optimization techniques including **ONNX Runtime Engine conversion, FP16 half-precision scaling, structural pruning (30%), and Dynamic INT8 Quantization**.

### 1. Stage 1: Detector (YOLOv8n) on GTSDB
* **Task**: Locate Traffic Signs

| Metric | FP32 Baseline (.pt) | Optimized Model (ONNX FP16) | Delta / Speedup |
| :--- | :---: | :---: | :---: |
| **Disk Size** | 5.99 MB | **6.01 MB** | ~0% |
| **mAP@50** | 96.52% | **96.22%** | -0.30% |
| **CPU Latency** | 378.35 ms | **272.22 ms** | **1.39x Speedup** |
| **GPU Latency** | 11.97 ms | **8.54 ms** | **1.40x Speedup** |

### 2. Stage 2: Classifier (MobileNetV3) on GTSRB
* **Task**: Fine-Grained 43-Class Classification

| Metric | FP32 Baseline (.pth) | Optimized Model (Dynamic INT8) | Delta / Speedup |
| :--- | :---: | :---: | :---: |
| **Disk Size** | 6.09 MB | **4.27 MB** | **-29.8% Size Reduction** |
| **Test Accuracy** | 97.90% | **97.82%** | -0.08% (Negligible) |
| **CPU Latency** | 10.35 ms | **10.36 ms** | 1.00x |
| **GPU Latency** | 5.77 ms | **10.18 ms** | (Dynamic INT8 runs on CPU) |

### 3. End-to-End Cascade Pipeline Latency
* Combined performance on a standard 1024x1024 high-resolution test frame:



```
=================== FINAL END-TO-END PIPELINE BENCHMARK ===================
Metric                           | CPU Pipeline           | GPU Pipeline          
----------------------------------------------------------------------------------
Execution Target                 | cpu                    | cuda:0                
ONNX Runtime Provider            | CPUExecutionProvider   | CUDAExecutionProvider 
----------------------------------------------------------------------------------
Stage 1 (YOLO ONNX) Latency      | 253.58              ms | 14.46               ms
Crop & Transform Latency         | 1.46                ms | 0.58                ms
Stage 2 Classifier Latency       | 11.93               ms | 6.22                ms
----------------------------------------------------------------------------------
Total End-to-End Latency         | 267.35  ± 8.55     ms | 21.56   ± 0.84     ms
End-to-End Throughput            | 3.7                 FPS | 46.4                FPS
==========================================================================
```

---


## 🛠️ Features Implemented

* **Dual Framework Integration**: Leveraged **TensorFlow/Keras** for initial prototyping, **PyTorch** for MobileNetV3 pipeline training, and **Ultralytics** for YOLOv8.
* **Post-Training Optimization**: 
  * Exported YOLOv8 weights to ONNX format with half-precision floating-point format (`FP16`).
  * Performed 30% structural unstructured pruning on MobileNetV3 weights followed by fine-tuning.
  * Implemented PyTorch dynamic quantization mapping model weights to `INT8` to save disk space and accelerate CPU inference.
* **Explainable AI (XAI)**: Integrated **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize and inspect activations on high-loss / misclassified sign samples to explain the classifier's decision-making.
* **Comprehensive Evaluation**: Metrics calculated include Accuracy, Precision, Recall, F1-Score, Multiply-Accumulate Operations (MACs), and hardware-specific latency breakdowns.

---

## 📂 Datasets Used

1. **GTSRB (German Traffic Sign Recognition Benchmark)**: Contains 39,209 training images and 12,630 testing images across 43 different traffic sign categories.
2. **GTSDB (German Traffic Sign Detection Benchmark)**: Full-resolution highway images containing annotated traffic signs for object detection.

# 
#
#
#
# 2: BMI and Sex Prediction from Spine X-ray Images

This repository contains a PyTorch-based deep learning pipeline designed to predict both **Body Mass Index (BMI)** (regression) and **Sex** (binary classification) simultaneously from spine X-ray images using multi-task learning.

## 📌 Project Overview
Predicting clinical and physiological indicators directly from medical imaging helps in non-invasive patient assessment. This project leverages transfer learning, robust data preprocessing, and validation frameworks to extract subtle anatomical patterns from spine X-rays.

## 🛠️ Key Features
- **Multi-Task Learning Architecture**: A unified network predicting categorical (Sex) and continuous (BMI) targets simultaneously using task-specific heads.
- **SOTA Architectures**: Features custom CNN baselines alongside transfer-learning setups utilizing pre-trained architectures like **ResNet-18**, **DenseNet-121**, and **EfficientNet-B0**.
- **Patient-Level Data Splitting**: Avoids data leakage by ensuring images from the same anonymized patient ID (`anon_id`) never span both train and validation splits.
- **Robust Validation**: Implements 5-Fold Cross-Validation ensembled predictions to combat variance and boost Kaggle benchmark performance.
- **Explainability (Grad-CAM)**: Integrates Gradient-weighted Class Activation Mapping to visualize and audit anatomical regions of interest driving classification decisions.

## 📂 Dataset Structure
- `train_images/` & `test_images/`: Raw X-ray scan files.
- `train_labels.csv`: Ground truth metadata including `anon_id`, `sex`, and `bmi`.
- `sample_submission.csv`: Target structure for hold-out predictions.

## 📈 Results Summary
- **K-Fold CV Average Accuracy (Sex)**: ~89.8%
- **K-Fold CV Average Loss (BMI)**: ~0.219 (Smooth L1 Loss)

## ⚖️ Deployment & Fairness Considerations
As analyzed within the notebook, immediate clinical deployment is recommended against until external multi-center validations are executed. Fairness audits indicated slight demographic performance variances, underscoring the need for further robust clinical auditing and uncertainty quantification.
