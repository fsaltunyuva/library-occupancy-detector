import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Load the pre-trained model from TensorFlow Hub
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Function to run object detection on an image
def run_detector(detector, image):
    converted_img = tf.image.convert_image_dtype(image, tf.uint8)[tf.newaxis, ...]
    result = detector(converted_img)
    return {key: value.numpy() for key, value in result.items()}

# Open a connection to the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_tensor = tf.convert_to_tensor(rgb_frame, dtype=tf.uint8)

    # Run the detector
    results = run_detector(detector, rgb_tensor)

    # Extract the results, removing the batch dimension
    boxes = results["detection_boxes"][0]  # shape: (100, 4)
    class_ids = results["detection_classes"][0]  # shape: (100,)
    scores = results["detection_scores"][0]  # shape: (100,)

    # Draw the detection results on the frame
    for i in range(len(scores)):
        score = scores[i]
        if score >= 0.5:  # Ensure we're working with a single score value
            box = boxes[i]
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])

            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = f"{int(class_ids[i])}: {score:.2f}"
            cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with the detection boxes
    cv2.imshow("Object Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
