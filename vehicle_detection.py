# ==============================
# Imports
# ==============================
import cv2
import numpy as np
from create_data import svc, x_scaler          # Pre-trained classifier (SVM) + feature scaler
from image_processing import extract_img_features  # Function to extract HOG/color features from images
from scipy.ndimage import label                # For labeling connected components in heatmaps
from tqdm import tqdm                          # Progress bar for video processing
from ultralytics import YOLO                   # Ultralytics YOLO for deep-learning detection


# ==============================
# Draw bounding boxes on an image
# ==============================
def draw_boxes(image, bbox, color=(0, 255, 0), thick=5):
    img = np.copy(image)
    for box in bbox:
        cv2.rectangle(img, box[0], box[1], color, thick)
    return img


# ==============================
# Sliding window generator
# ==============================
def create_search_window(image, x_start_stop=[None, None], y_start_stop=[None, None],
                         xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # Default values: full image in x, region of interest in y
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = image.shape[1]  # image width
    if y_start_stop[0] is None:
        y_start_stop[0] = 350             # horizon line (ignore sky)
    if y_start_stop[1] is None:
        y_start_stop[1] = image.shape[0]  # image height

    # Span of search region
    x_span = x_start_stop[1] - x_start_stop[0]
    y_span = y_start_stop[1] - y_start_stop[0]

    # Step sizes based on overlap
    window_stride_x = xy_window[0] * (1 - xy_overlap[0])
    window_stride_y = xy_window[1] * (1 - xy_overlap[1])

    # Number of windows in x and y
    nwindows_x = np.int32(((x_span - xy_window[0]) / window_stride_x) + 1)
    nwindows_y = np.int32(((y_span - xy_window[1]) / window_stride_y) + 1)

    # Generate windows
    windows_list = []
    for y in range(nwindows_y):
        for x in range(nwindows_x):
            start_x = np.int32(x_start_stop[0] + x * window_stride_x)
            end_x = np.int32(start_x + xy_window[0])
            start_y = np.int32(y_start_stop[0] + y * window_stride_y)
            end_y = np.int32(start_y + xy_window[1])
            windows_list.append(((start_x, start_y), (end_x, end_y)))

    return windows_list


# ==============================
# Multi-scale search windows
# ==============================
def generate_all_windows(image):
    h, w, c = image.shape
    all_windows = []

    # Small windows (far cars)
    all_windows += create_search_window(image,
                                        x_start_stop=[200, w - 200],
                                        y_start_stop=[360, 500],
                                        xy_window=(64, 64),
                                        xy_overlap=(0.75, 0.75))

    # Medium windows
    all_windows += create_search_window(image,
                                        x_start_stop=[100, w - 100],
                                        y_start_stop=[380, 600],
                                        xy_window=(96, 96),
                                        xy_overlap=(0.75, 0.75))

    # Large windows
    all_windows += create_search_window(image,
                                        x_start_stop=[None, None],
                                        y_start_stop=[400, 656],
                                        xy_window=(128, 128),
                                        xy_overlap=(0.75, 0.75))

    # Extra-large windows (very close cars)
    all_windows += create_search_window(image,
                                        x_start_stop=[None, None],
                                        y_start_stop=[420, 656],
                                        xy_window=(160, 160),
                                        xy_overlap=(0.75, 0.75))

    print("Generated windows:", len(all_windows))
    return all_windows


# ==============================
# Heatmap utilities
# ==============================
def add_heat(heatmap, bbox_list):
    # Add +1 for each bounding box region
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    # Suppress low-heat pixels (false positives)
    heatmap[heatmap <= threshold] = 0
    return heatmap


# ==============================
# Classical sliding window detection with SVM
# ==============================
def search_img(image, svc, X_scaler, threshold=2, dilate_kernel=15):
    hot_boxes = []
    window_boxes = generate_all_windows(image)

    # Step 1: classify each window using SVM
    for window in window_boxes:
        test_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = extract_img_features(test_img)
        scaled_features = X_scaler.transform(np.array(features).reshape(1, -1))
        pred = svc.predict(scaled_features)
        if pred == 1:  # Vehicle detected
            hot_boxes.append(window)

    # Step 2: build heatmap
    heatmap = np.zeros_like(image[:, :, 0]).astype(np.float32)
    heatmap = add_heat(heatmap, hot_boxes)

    # Step 3: threshold heatmap
    heatmap = apply_threshold(heatmap, threshold)

    # Step 4: dilate to merge nearby detections
    kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    heatmap = cv2.dilate(heatmap, kernel, iterations=1)

    # Step 5: label connected components (cars)
    labels = label(heatmap)

    return hot_boxes, labels


# ==============================
# Draw bounding boxes from labels
# ==============================
def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1] + 1):  # Loop over detected cars
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 6)
    return img


# ==============================
# Process a single frame
# ==============================
def process_frame(frame, svc, x_scaler, heat_threshold=2, dilate_kernel=15):
    hot_windows = []
    window_boxes = generate_all_windows(frame)

    # Classify windows with SVM
    for window in window_boxes:
        test_img = cv2.resize(frame[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = extract_img_features(test_img)
        scaled_features = x_scaler.transform(np.array(features).reshape(1, -1))
        pred = svc.predict(scaled_features)

        if pred == 1:
            hot_windows.append(window)

    # Build + threshold heatmap
    heatmap = np.zeros_like(frame[:, :, 0]).astype(np.float32)
    heatmap = add_heat(heatmap, hot_windows)
    heatmap = apply_threshold(heatmap, heat_threshold)

    # Dilate blobs
    kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    heatmap = cv2.dilate(heatmap, kernel, iterations=1)

    # Label and draw boxes
    labels = label(heatmap)
    output = draw_labeled_bboxes(np.copy(frame), labels)

    return output


# ==============================
# Vehicle detection on full video (classical SVM)
# ==============================
def vehicle_detection(input_path, output_path, svc, x_scaler):
    cap = cv2.VideoCapture(input_path)

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")

    # Frame-by-frame processing
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame, svc, x_scaler,
                                        heat_threshold=2,
                                        dilate_kernel=15)

        out.write(processed_frame)
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"✅ Done! Processed video saved to: {output_path}")


# ==============================
# Vehicle detection with YOLOv11 (deep learning)
# ==============================
def vehicle_detection_YOLO(yolo_model="yolo11n.pt",
                           input_stream='project_video.mp4',
                           output_destination='YOLO_out.mp4'):
    model = YOLO(yolo_model)
    result = model(input_stream, save=True, show=True, name=output_destination)
    return result


# ==============================
# Run both pipelines
# ==============================
vehicle_detection('project_video.mp4', 'VD_final_out.mp4', svc, x_scaler)  # Classical SVM pipeline
vehicle_detection_YOLO("yolo11n.pt", 'project_video.mp4')                   # YOLO deep-learning pipeline
