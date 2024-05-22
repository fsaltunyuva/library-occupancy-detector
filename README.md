# Library Occupancy Detector

This project aims to develop a computer vision model to detect seat/desk occupancy
within libraries and show the occupancy rate of the libraries to students. The model will use
computer vision techniques to identify unoccupied tables, chairs, and other seating areas within
the TEDU library environment. The primary objective of this project is to enhance the study
experience for students by providing real-time information on the availability of the library. With
the assistance of our computer vision model, students will be able to quickly inform about
unoccupied places.

The project will run on the camera installed in the library, and our model will instantly
make an inference about the occupancy rate at the table through the data obtained from the
camera. Our model will also be able to detect situations such as items left on the table or students
who do not have anything on the table but are in their chairs.

## Followed Documentations - Tutorials - Repositories

* https://github.com/ultralytics/ultralytics

* https://github.com/BenGreenfield825/Tensorflow-Object-Detection-with-Tensorflow-2.0/blob/master/detection_scripts/detect_from_webcam.py

* https://www.tensorflow.org/hub/tutorials/object_detection?hl=tr

* https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993

## Important Notes

* Do not use names like "tensorflow.py" or "tensorflow_webcam.py" that clashes with TensorFlow's internal modules, it can cause import issues.


## Required Packages

### For TensorFlow

`pip install tensorflow_hub`

`pip install tensorflow`

### For YOLO

`pip install ultralytics`
