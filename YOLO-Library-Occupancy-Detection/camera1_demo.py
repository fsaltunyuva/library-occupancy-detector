from ultralytics import YOLO
import cv2
import math
import firebase_admin
from firebase_admin import credentials, db

# Firebase Admin SDK JSON credentials file
cred = credentials.Certificate("path-to-json-file")

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://lod-db-default-rtdb.firebaseio.com/'  # Replace with your database URL
})

# Reference to the database path you want to update
# TODO: Create another database reference for the demo
ref = db.reference('occupancy/Camera1')

# Start webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 0 for the embedded webcam, 1 for the second, 2 for the third, etc.

# Webcam Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)

# Load model
# TODO: Upgrade the YOLO to 11.0 version for better performance (Use the proper one in the link located in README.md)
model = YOLO("yolov10n.pt")  # s-m-b-l-x models can be used for further accuracy
# (https://github.com/THU-MIG/yolov10?tab=readme-ov-file#performance)

# model.to('cuda')  # TODO: Use GPU for faster inference

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
              "teddy bear", "hair drier", "toothbrush"]

# model.names can be used to get the class names of the current YOLO version

# model.to('cuda')  # Use GPU for faster inference

objects_that_can_be_detected = ["cell phone", "bottle", "backpack", "umbrella", "handbag", "suitcase", "apple",
                                "skateboard", "cup", "fork", "knife", "spoon", "bowl", "sandwich",
                                "orange", "brocoli", "carrot", "hot dog", "pizza", "donut", "cake",
                                "laptop", "mouse", "keyboard", "book", "scissors"]  # Add other objects if needed

# Define the specific area (ROI - Region of Interest) for the chair
# For example, the chair is located at the region (x1, y1) to (x2, y2)
chair_roi1 = (0, 30, 340, 610)  # (x1, y1, x2, y2)
chair_roi2 = (350, 30, 680, 610)  # (x1, y1, x2, y2)
chair_roi3 = (690, 30, 1024, 610)  # (x1, y1, x2, y2)

while True:
    chair1_occupancy = False
    chair2_occupancy = False
    chair3_occupancy = False
    chair1_hold = False
    chair2_hold = False
    chair3_hold = False

    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)

    cell_phone_detected_for_chair1 = False
    cell_phone_detected_for_chair2 = False
    cell_phone_detected_for_chair3 = False
    person_detected_for_chair1 = False
    person_detected_for_chair2 = False
    person_detected_for_chair3 = False

    for result in results:
        boxes = result.boxes

        for box in boxes:
            # Class name
            cls = int(box.cls[0]) # ?????

            # Check if the detected object is either a bottle or a person
            if classNames[cls] == "person":
                # Bounding box
                x1_person, y1_person, x2_person, y2_person = box.xyxy[0]
                x1_person, y1_person, x2_person, y2_person = int(x1_person), int(y1_person), int(x2_person), int(y2_person)

                # Draw bounding box for person
                cv2.rectangle(img, (x1_person, y1_person), (x2_person, y2_person), (0, 255, 255), 3)

                # Confidence
                confidence_person = math.ceil((box.conf[0]*100)) / 100

                # Add confidence to the top right corner of the box and adjust the text settings
                text_person = f"Person {confidence_person}"
                text_size_person, _ = cv2.getTextSize(text_person, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                text_x_person = x2_person - text_size_person[0]
                text_y_person = y1_person - 10 if y1_person - 10 > 10 else y1_person + 10  # Adjust text position if it's too close to the top

                cv2.putText(img, text_person, (text_x_person, text_y_person), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                # TODO: Check intersection areas in a for loop in the case of unspecified chair count (Currently only 2 chairs are considered)
                # Check if the person's bounding box overlaps with the specified area
                intersection_area_of_chair1 = max(0, min(x2_person, chair_roi1[2]) - max(x1_person, chair_roi1[0])) * max(
                    0, min(y2_person, chair_roi1[3]) - max(y1_person, chair_roi1[1]))

                intersection_area_of_chair2 = max(0, min(x2_person, chair_roi2[2]) - max(x1_person, chair_roi2[0])) * max(
                    0, min(y2_person, chair_roi2[3]) - max(y1_person, chair_roi2[1]))

                intersection_area_of_chair3 = max(0, min(x2_person, chair_roi3[2]) - max(x1_person, chair_roi3[0])) * max(
                    0, min(y2_person, chair_roi3[3]) - max(y1_person, chair_roi3[1]))

                person_area = (x2_person - x1_person) * (y2_person - y1_person)

                if intersection_area_of_chair1 / person_area >= 0.6:
                    person_detected_for_chair1 = True

                if intersection_area_of_chair2 / person_area >= 0.6:
                    person_detected_for_chair2 = True

                if intersection_area_of_chair3 / person_area >= 0.6:
                    person_detected_for_chair3 = True

            # elif classNames[cls] == "cell phone":
            elif objects_that_can_be_detected.__contains__(classNames[cls]): # Check if the detected object is in the list of objects that can be detected
                # TODO: Change the variables and displayed name for the object detected
                # Bounding box
                x1_cell_phone, y1_cell_phone, x2_cell_phone, y2_cell_phone = box.xyxy[0]
                x1_cell_phone, y1_cell_phone, x2_cell_phone, y2_cell_phone = int(x1_cell_phone), int(y1_cell_phone), int(x2_cell_phone), int(y2_cell_phone)

                # Draw bounding box for cell phone
                cv2.rectangle(img, (x1_cell_phone, y1_cell_phone), (x2_cell_phone, y2_cell_phone), (255, 255, 0), 3)

                # Confidence
                confidence_cell_phone = math.ceil((box.conf[0] * 100)) / 100

                # Add confidence to the top right corner of the box
                text_bottle = f"cell phone {confidence_cell_phone}"
                text_size_cell_phone, _ = cv2.getTextSize(text_bottle, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                text_x_cell_phone = x2_cell_phone - text_size_cell_phone[0]
                text_y_cell_phone = y1_cell_phone - 10 if y1_cell_phone - 10 > 10 else y1_cell_phone + 10  # Adjust text position if it's too close to the top

                cv2.putText(img, text_bottle, (text_x_cell_phone, text_y_cell_phone), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

                # TODO: Check intersection areas in a for loop in the case of unspecified chair count (Currently only 2 chairs are considered)
                # Check if the bottle's bounding box overlaps with the specified area
                intersection_area_of_chair1 = max(0, min(x2_cell_phone, chair_roi1[2]) - max(x1_cell_phone, chair_roi1[0])) * max(
                    0, min(y2_cell_phone, chair_roi1[3]) - max(y1_cell_phone, chair_roi1[1]))

                intersection_area_of_chair2 = max(0, min(x2_cell_phone, chair_roi2[2]) - max(x1_cell_phone, chair_roi2[0])) * max(
                    0, min(y2_cell_phone, chair_roi2[3]) - max(y1_cell_phone, chair_roi2[1]))

                intersection_area_of_chair3 = max(0, min(x2_cell_phone, chair_roi3[2]) - max(x1_cell_phone, chair_roi3[0])) * max(
                    0, min(y2_cell_phone, chair_roi3[3]) - max(y1_cell_phone, chair_roi3[1]))

                cell_phone_area = (x2_cell_phone - x1_cell_phone) * (y2_cell_phone - y1_cell_phone)

                if intersection_area_of_chair1 / cell_phone_area >= 0.6:
                    cell_phone_detected_for_chair1 = True

                if intersection_area_of_chair2 / cell_phone_area >= 0.6:
                    cell_phone_detected_for_chair2 = True

                if intersection_area_of_chair3 / cell_phone_area >= 0.6:
                    cell_phone_detected_for_chair3 = True

            # TODO: Create other elif statements for other objects if needed

    # Display a message if the chair is occupied by a person or a bottle
    # (Print hold if the chair is empty and an object is detected)

    # For chair 1
    if person_detected_for_chair1:
        cv2.putText(img, "Chair 1 Occupied", (chair_roi1[0], chair_roi1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        chair1_occupancy = True

    elif cell_phone_detected_for_chair1 and not person_detected_for_chair1:
        cv2.putText(img, "Chair 1 Hold", (chair_roi1[0], chair_roi1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        chair1_hold = True

    # For chair 2
    if person_detected_for_chair2:
        cv2.putText(img, "Chair 2 Occupied", (chair_roi2[0], chair_roi2[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        chair2_occupancy = True

    elif cell_phone_detected_for_chair2 and not person_detected_for_chair2:
        cv2.putText(img, "Chair 2 Hold", (chair_roi2[0], chair_roi2[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        chair2_hold = True

    # For chair 3
    if person_detected_for_chair3:
        cv2.putText(img, "Chair 3 Occupied", (chair_roi3[0], chair_roi3[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        chair3_occupancy = True

    elif cell_phone_detected_for_chair3 and not person_detected_for_chair3:
        cv2.putText(img, "Chair 3 Hold", (chair_roi3[0], chair_roi3[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        chair3_hold = True

    # Draw the ROI (Region of Interest) for the chair
    cv2.rectangle(img, (chair_roi1[0], chair_roi1[1]), (chair_roi1[2], chair_roi1[3]), (0, 255, 0), 2)
    cv2.rectangle(img, (chair_roi2[0], chair_roi2[1]), (chair_roi2[2], chair_roi2[3]), (0, 255, 0), 2)
    cv2.rectangle(img, (chair_roi3[0], chair_roi3[1]), (chair_roi3[2], chair_roi3[3]), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

    # time.sleep(1)

    ref.child('chair1').update({
        "occupied": chair1_occupancy,
        "hold": chair1_hold
    })
    ref.child('chair2').update({
        "occupied": chair2_occupancy,
        "hold": chair2_hold
    })
    ref.child('chair3').update({
        "occupied": chair3_occupancy,
        "hold": chair3_hold
    })
    print("Database updated successfully!")

cap.release()
cv2.destroyAllWindows()