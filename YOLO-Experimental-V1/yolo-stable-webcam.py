from ultralytics import YOLO
import cv2
import math

# Start webcam
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
chair_roi = (0, 30, 310, 610)  # (x1, y1, x2, y2)
chair_roi2 = (310, 30, 630, 610)  # (x1, y1, x2, y2)

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)

    chair_occupied = False
    chair_occupied2 = False
    person_detected = False
    person_detected2 = False

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Class name
            cls = int(box.cls[0])

            # Check if the detected object is either a bottle or a person
            if classNames[cls] == "person":
                # Bounding box
                x1_person, y1_person, x2_person, y2_person = box.xyxy[0]
                x1_person, y1_person, x2_person, y2_person = int(x1_person), int(y1_person), int(x2_person), int(y2_person)

                # Draw bounding box for person
                cv2.rectangle(img, (x1_person, y1_person), (x2_person, y2_person), (0, 255, 255), 3)

                # Confidence
                confidence_person = math.ceil((box.conf[0]*100))/100

                # Add confidence to the top right corner of the box
                text_person = f"Person {confidence_person}"
                text_size_person, _ = cv2.getTextSize(text_person, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                text_x_person = x2_person - text_size_person[0]
                text_y_person = y1_person - 10 if y1_person - 10 > 10 else y1_person + 10  # Adjust text position if it's too close to the top

                cv2.putText(img, text_person, (text_x_person, text_y_person), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                # Check if the person's bounding box overlaps with the specified area
                intersection_area = max(0, min(x2_person, chair_roi[2]) - max(x1_person, chair_roi[0])) * max(
                    0, min(y2_person, chair_roi[3]) - max(y1_person, chair_roi[1]))

                intersection_area2 = max(0, min(x2_person, chair_roi2[2]) - max(x1_person, chair_roi2[0])) * max(
                    0, min(y2_person, chair_roi2[3]) - max(y1_person, chair_roi2[1]))

                person_area = (x2_person - x1_person) * (y2_person - y1_person)
                if intersection_area / person_area >= 0.6:
                    person_detected = True

                if intersection_area2 / person_area >= 0.6:
                    person_detected2 = True


            elif classNames[cls] == "cell phone":
                # Bounding box
                x1_bottle, y1_bottle, x2_bottle, y2_bottle = box.xyxy[0]
                x1_bottle, y1_bottle, x2_bottle, y2_bottle = int(x1_bottle), int(y1_bottle), int(x2_bottle), int(y2_bottle)

                # Draw bounding box for bottle
                cv2.rectangle(img, (x1_bottle, y1_bottle), (x2_bottle, y2_bottle), (255, 255, 0), 3)

                # Confidence
                confidence_bottle = math.ceil((box.conf[0]*100))/100

                # Add confidence to the top right corner of the box
                text_bottle = f"cell phone {confidence_bottle}"
                text_size_bottle, _ = cv2.getTextSize(text_bottle, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                text_x_bottle = x2_bottle - text_size_bottle[0]
                text_y_bottle = y1_bottle - 10 if y1_bottle - 10 > 10 else y1_bottle + 10  # Adjust text position if it's too close to the top

                cv2.putText(img, text_bottle, (text_x_bottle, text_y_bottle), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

                # Check if the bottle's bounding box overlaps with the specified area
                intersection_area = max(0, min(x2_bottle, chair_roi[2]) - max(x1_bottle, chair_roi[0])) * max(
                    0, min(y2_bottle, chair_roi[3]) - max(y1_bottle, chair_roi[1]))

                intersection_area2 = max(0, min(x2_bottle, chair_roi2[2]) - max(x1_bottle, chair_roi2[0])) * max(
                    0, min(y2_bottle, chair_roi2[3]) - max(y1_bottle, chair_roi2[1]))
                bottle_area = (x2_bottle - x1_bottle) * (y2_bottle - y1_bottle)
                if intersection_area / bottle_area >= 0.6:
                    chair_occupied = True

                if intersection_area2 / bottle_area >= 0.6:
                    chair_occupied2 = True



    # Display a message if the chair is occupied by a person or a bottle
    if chair_occupied or person_detected:
        # cv2.rectangle(img, (chair_roi[0], chair_roi[1]),
        #               (chair_roi[0] + chair_roi[2], chair_roi[1] + chair_roi[3]),
        #               (0, 0, 255), 2)
        #
        # # Metni ekleyin
        cv2.putText(img, "Chair 1 Occupied", (chair_roi[0], chair_roi[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    if chair_occupied2 or person_detected2:
        # cv2.rectangle(img, (chair_roi2[0], chair_roi2[1]),
        #               (chair_roi2[0] + chair_roi2[2], chair_roi2[1] + chair_roi2[3]),
        #               (0, 0, 255), 2)
        #
        # # Metni ekleyin
        cv2.putText(img, "Chair 2 Occupied", (chair_roi2[0], chair_roi2[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # Draw the ROI for the chair
    cv2.rectangle(img, (chair_roi[0], chair_roi[1]), (chair_roi[2], chair_roi[3]), (0, 255, 0), 2)
    cv2.rectangle(img, (chair_roi2[0], chair_roi2[1]), (chair_roi2[2], chair_roi2[3]), (0, 255, 0), 2)
    # Display the frame
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
