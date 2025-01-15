# Library Occupancy Detector

[Project's Website](https://library-occupancy-detector-l5dr64zqg-burakkoc5s-projects.vercel.app)

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

## Active TODOs

### Code Related TODOs
- [ ] Using GPU for better performance.
- [x] Upgrading to YOLOv10 for better results.
- [x] Upgrading to YOLOv11 for better results.
- [ ] Making the code more generic for multiple chairs.
- [x] For better performance and workload, detect objects every second instead of every frame.
- [ ] Detect other possible objects that may be used for holding (backpacks, bottle, umbrella, book, etc.)
- [x] Send occupancy data to Firebase database.
- [ ] Detect other objects that are not included in class names of YOLO.

### Project Related TODOs
- [x] Testing the cameras with USB extension cables.
- [x] Installation of the cameras in the library.
- [x] Customize the script for the library environment after camera installation.
- [x] Develop the desktop application for the project in Flutter.
- [ ] Create a website for the project.

## Followed Documentations - Tutorials - Repositories

* [Ultralytics GitHub Repository](https://github.com/ultralytics/ultralytics)

* [Sample Code for Webcam Object Detection](https://github.com/BenGreenfield825/Tensorflow-Object-Detection-with-Tensorflow-2.0/blob/master/detection_scripts/detect_from_webcam.py)

* [TensorFlow Object Detection Tutorial](https://www.tensorflow.org/hub/tutorials/object_detection)

* [Classnames for the YOLO](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)

* [YOLO's datasets for specified tasks](https://docs.ultralytics.com/models/yolo11/#supported-tasks-and-modes)

* [TensorFlow and CUDA Compatibility](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)

* [CPU to CUDA](https://github.com/ultralytics/ultralytics/issues/3084)
    * [Another Source](https://stackoverflow.com/questions/75691440/how-to-move-yolov8-model-onto-gpu)

> [!IMPORTANT]
> * Do not use names like "tensorflow.py" or "tensorflow_webcam.py" that clashes with TensorFlow's internal modules, it can cause import issues.
> * [YOLO Recommends to run the script on GPUs with a minimum of 8GB of memory.](https://docs.ultralytics.com/help/FAQ/)

## Required Packages

### For TensorFlow
```
pip install tensorflow
```

### For YOLO
```
pip install ultralytics
```

## Poster

<p float="left">
<img src="https://github.com/fsaltunyuva/library-occupancy-detector/blob/main/Documents/Poster.png"/>  
</p>
