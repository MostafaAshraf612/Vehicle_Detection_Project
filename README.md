# 🚘 Vehicle Detection Project  
**Udacity Self-Driving Car Nanodegree (v1.0)**  
**Developer:** [Mostafa Ashraf El Sayed](https://www.linkedin.com/in/mostafa-ashraf-612)

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)  
![Language: Python](https://img.shields.io/badge/Language-Python3-blue.svg)  
![Status: Completed](https://img.shields.io/badge/Status-Completed-success.svg)

---

## 📚 Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Architecture](#architecture)  
- [Repository Structure](#repository-structure)  
- [Setup & Execution](#setup--execution)  
- [Algorithm Summary](#algorithm-summary)  
- [Comparison](#comparison)  
- [Results & Demonstration](#results--demonstration)  
- [License](#license)  
- [Contact](#contact)

---

## 📌 Overview

A dual-pipeline vehicle detection system comparing traditional computer vision techniques with deep learning models.  
This project simulates perception modules for autonomous driving using both **SVM + HOG** and **YOLOv8**.

Developed as part of the **Udacity Self-Driving Car Nanodegree**, it showcases object detection strategies for real-time vehicle localization.

---

## ✨ Features

- **HOG + SVM Classifier:** Detects vehicles using handcrafted features and sliding window search  
- **YOLOv8 Detector:** Real-time, end-to-end deep learning model for multi-vehicle detection  
- **Video Pipeline:** Processes driving footage frame-by-frame  
- **Modular Design:** Easy to switch between detection methods  
- **Performance Comparison:** Benchmarks accuracy and speed across both pipelines

---

## 🧠 Architecture

1. **Data Preparation** – Extracts labeled car/non-car images  
2. **Feature Extraction** – Applies HOG, color histograms, and spatial binning  
3. **SVM Training** – Builds a binary classifier for vehicle detection  
4. **YOLOv8 Inference** – Loads pretrained model for real-time detection  
5. **Video Processing** – Applies detection frame-by-frame and overlays results

---

## 📁 Repository Structure
```
Vehicle_Detection_Project/  
├── asset/  
│   ├── Yolo_demo.gif               # Yolo pipeline demo
│   ├── SVM_demo.gif                # SVM pipeline demo
│   ├── project_video_svm2.mp4      # SVM pipeline full video
│   └── project_video_yolo.avi      # YOLOv8 pipeline full video
├── src/  
│   ├── create_data.py              # Dataset creation and preprocessing  
│   ├── image_processing.py         # Feature extraction and filtering  
│   ├── vehicle_detection.py        # Main detection runner  
├── README.md  
└── test_images.zip                 # Sample test images
```
---

## 🛠️ Setup & Execution

### 🔧 Requirements

- Python 3.x  
- OpenCV  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Ultralytics (for YOLOv8)

### 📦 Steps

#### 🔧 **Step 1: Clone the Repository**

```bash
git clone https://github.com/MostafaAshraf612/Vehicle_Detection_Project.git
cd Vehicle_Detection_Project
```
#### 🔧 ** Step 2: Install Ultralytics for yolo**

```bash
pip install ultralytics
```
#### 🔧 **  Step 3: Run the Detection Pipelines**

```bash
# Run SVM pipeline
python src/vehicle_detection.py // choose the vehicle_detection function

# Run YOLOv8 pipeline
python src/vehicle_detection.py // choose the vehicle_detection_YOLO function
```
---
## 📈 Algorithm Summary

The project compares two detection strategies:

- **SVM + HOG:** Uses handcrafted features and sliding window search  
- **YOLOv8:** Deep learning model trained on COCO dataset for real-time detection  
Each frame is processed to detect vehicles, and results are visualized with bounding boxes.

---

## ⚖️ Comparison

| Feature               | SVM + HOG                         | YOLOv8                              |
|----------------------|-----------------------------------|-------------------------------------|
| Accuracy             | ~98.4%                            | Very High (pretrained on COCO)      |
| Speed                | Slower (frame-by-frame)           | Real-time                           |
| Complexity           | Manual feature engineering        | End-to-end learning                 |
| False Positives      | Requires heatmap filtering        | Minimal with confidence threshold   |
| Deployment           | Lightweight, CPU-friendly         | GPU recommended for real-time       |

---

## 🎥 Results & Demonstration

The system successfully detects vehicles in varied traffic scenes using both pipelines.

📹 **Demo Preview:**  
## 🎥 Pipeline Demonstrations

| SVM + HOG Detection | YOLOv8 Detection |
|---------------------|------------------|
| <img src="https://github.com/MostafaAshraf612/Vehicle_Detection_Project/blob/main/asset/SVM_demo.gif" width="400"/> | <img src="https://github.com/MostafaAshraf612/Vehicle_Detection_Project/blob/main/asset/Yolo_demo.gif" width="400"/> |

---

### ✅ Performance Metrics

| 🔍 **Metric**              | 📊 **SVM**         | ⚡ **YOLOv8**       |
|---------------------------|-------------------|--------------------|
| Detection Accuracy        | ~98.4%            | Very High          |
| Frame Processing Time     | ~0.5 sec/frame    | ~0.05 sec/frame    |
| Multi-Vehicle Detection   | Limited           | Strong             |
| Robustness to Lighting    | Moderate          | Strong             |
| Real-Time Capability      | No                | Yes                |

---

## 📄 License

This project is released under the **[MIT License](LICENSE)**.

---

## 📬 Contact

For technical inquiries or collaboration opportunities:

**Mostafa Ashraf El Sayed**  
🔗 [LinkedIn](https://www.linkedin.com/in/mostafa-ashraf-612)  
💻 [GitHub](https://github.com/MostafaAshraf612)  
📧 [mostafashrafelsayed612@gmail.com](mailto:mostafashrafelsayed612@gmail.com)
