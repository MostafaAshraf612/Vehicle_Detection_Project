# 🚘 Vehicle Detection Project

A simulation-based comparison of traditional computer vision and deep learning techniques for vehicle detection in autonomous driving contexts.

## 📌 Project Summary

This project explores two approaches to vehicle detection:

- **Traditional CV + SVM**  
  Feature extraction using HOG, spatial binning, and color histograms, followed by a Linear SVM classifier for car/non-car classification.

- **Deep Learning with YOLOv11**  
  Real-time object detection using a modern neural network architecture, trained end-to-end for vehicle localization.

## 🎯 Objectives

- Compare handcrafted feature pipelines vs. deep learning models  
- Evaluate detection performance on images and video  
- Simulate perception modules for autonomous vehicle applications

## 🧠 Techniques Used

| Approach       | Tools & Methods                                      |
|----------------|------------------------------------------------------|
| Traditional CV | HOG, Color Histograms, Spatial Binning, Linear SVM   |
| Deep Learning  | YOLOv8, OpenCV, Python                               |

## 📂 Repo Structure
```
Vehicle_Detection_Project/
├── asset/
│   ├── project_video_svm2.mp4
│   └── project_video_yolo.avi
├── src/
│   ├── create_data.py
│   ├── image_processing.py
│   ├── vehicle_detection.py
├── README.md
└── test_images.zip
```
## 📈 Results

### SVM Classifier

- Achieved ~98.4% accuracy on the test dataset (car vs. non-car classification)
- Detects vehicles in images and video streams
- Performance tunable via window size and heatmap threshold
- Demo video:"assets/project_video_svm.mp4"


### YOLOv8 Detector

- Real-time vehicle detection with high confidence
- Capable of detecting multiple vehicles simultaneously
- Suitable for perception modules in autonomous driving systems
- Demo video: "assets/project_video_yolo.avi"


---

> 💡 Developed as part of the Udacity Self Driving Car Nanodegree(v1.0)..  
> This project showcases simulation-based perception techniques for autonomous vehicles.
