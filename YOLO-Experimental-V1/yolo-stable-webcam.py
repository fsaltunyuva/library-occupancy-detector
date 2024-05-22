from ultralytics import YOLO
import cv2
import math

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Define the specific area (ROI) for the chair
# For example, the chair is located at the region (x1, y1) to (x2, y2)
chair_roi = (200, 150, 400, 300)  # (x1, y1, x2, y2)

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)

    chair_occupied = False

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence
            confidence = math.ceil((box.conf[0]*100))/100

            # Class name
            cls = int(box.cls[0])

            # Add confidence to the top right corner of the box
            text = f"{classNames[cls]} {confidence}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            text_x = x2 - text_size[0]
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10  # Adjust text position if it's too close to the top

            cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Check if the detected object is within the ROI
            if classNames[cls] == "person":
                if (x1 < chair_roi[2] and x2 > chair_roi[0] and
                        y1 < chair_roi[3] and y2 > chair_roi[1]):
                    chair_occupied = True

    # Draw the ROI for the chair
    cv2.rectangle(img, (chair_roi[0], chair_roi[1]), (chair_roi[2], chair_roi[3]), (0, 255, 0), 2)

    # Display a message if the chair is occupied
    if chair_occupied:
        cv2.putText(img, "Chair Occupied", (chair_roi[0], chair_roi[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
