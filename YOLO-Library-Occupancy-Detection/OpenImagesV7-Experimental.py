from ultralytics import YOLO
import cv2

# Load an Open Images Dataset V7 pretrained YOLOv8n model
model = YOLO("yolov8x-oiv7.pt")

# model = YOLO("yolov10x.pt")

# Run prediction
results = model.predict(source=r"path-to-image",conf=0.07, save=True, show=True)

# Print details of detections
for result in results:
    print("Bounding Box Coordinates (xmin, ymin, xmax, ymax):", result.boxes.xyxy if result.boxes else "No boxes")
    print("Class Indices:", result.boxes.cls if result.boxes else "No classes")
    print("Confidence Scores:", result.boxes.conf if result.boxes else "No confidence scores")