from ultralytics import YOLO
import cv2

#
# # Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
#
# # Use the model
# model.train(data="coco8.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Start webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

result, image = cam.read()

# Define path to the image file
source = "C:\\Users\\Lenovo\\Downloads\\bus.jpg"

# Run inference on the source
results = model(image)  # list of Results objects

print(results)
