## Overview  

This project implements a vehicle detection pipeline using both **traditional computer vision techniques** and **modern deep learning methods**.  

1. **Feature Extraction & SVM Classifier**  
   - Uses Histogram of Oriented Gradients (HOG), color histograms, and spatial binning.  
   - Features are scaled and fed into a **Linear SVM** for car vs. non-car classification.  

2. **Deep Learning with YOLO**  
   - Explores the YOLOv8 model for real-time object detection.  
   - Provides a comparison between traditional methods and deep learning performance.  

3. **Applications**  
   - Detecting vehicles in still images.  
   - Extending to video streams for self-driving car perception.  

This repo contains **data preprocessing utilities, feature extraction modules, training scripts, and evaluation notebooks**, making it a complete framework for experimenting with vehicle detection.  
## How It Works  

The project is divided into two main approaches for vehicle detection:  

### 1. Traditional Computer Vision + Machine Learning  
- Images are processed using **feature extraction techniques**:  
  - **Spatial Binning** – reduces image size and flattens pixel values.  
  - **Color Histograms** – captures color distribution across channels.  
  - **HOG (Histogram of Oriented Gradients)** – extracts edge and texture patterns.  
- The extracted features are scaled and combined into a single feature vector.  
- A **Linear SVM classifier** is trained to distinguish between car and non-car images.  

### 2. Deep Learning with YOLO  
- Integrates **YOLOv8**, a modern object detection network.  
- Unlike handcrafted features, YOLO learns features directly from data.  
- Provides **real-time detection** performance, making it suitable for self-driving applications.  

### Goal  
The purpose of this project is to **compare traditional and deep learning methods** for vehicle detection, highlighting their strengths and trade-offs in terms of accuracy, speed, and complexity.  

## Results  

### SVM Classifier  
- Achieved ~98.4% accuracy on the test dataset (cars vs. non-cars).
- Example detections:
<img width="960" height="443" alt="image" src="https://github.com/user-attachments/assets/60b55de6-401f-464a-be76-c1f2abd1f0bc" />
 -- You Can watch the full result video named "VD_Final_out.mp4"

### YOLOv11 Detector  
- Provides **real-time vehicle detection** on images and video.  
- Detects multiple vehicles simultaneously with high confidence.  
- Example detections:  
<img width="960" height="443" alt="image" src="https://github.com/user-attachments/assets/e8856527-2605-4f22-bc59-b1d2a8b4bad3" />
 -- You Can watch the full result video named "YOLO_out.mp4"





