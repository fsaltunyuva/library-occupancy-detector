from ultralytics import YOLO

# Load a model
# model = YOLO("yolov10n.pt")  # load a pretrained model (recommended for training)
# Use pretrained LVIS model to predict
model = YOLO("yolov8n.pt")

results = model.predict(source="C:\\Users\\furkan\\Pictures\\Camera Roll\\WIN_20241118_12_19_44_Pro.jpg", save=True, show=True)


# Print details of detections
for result in results:
    print("Bounding Box Coordinates (xmin, ymin, xmax, ymax):", result.boxes.xyxy if result.boxes else "No boxes")
    print("Class Indices:", result.boxes.cls if result.boxes else "No classes")
    print("Confidence Scores:", result.boxes.conf if result.boxes else "No confidence scores")

# Train the model
# results = model.train(data="lvis.yaml", epochs=100, imgsz=640)