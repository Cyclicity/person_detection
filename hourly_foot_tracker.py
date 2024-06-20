import cv2
import time
from datetime import datetime

# Load class names from coco.names file into a list
classNames = []
classFile = "/home/pi/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Paths to the configuration and weights files for the SSD MobileNet model
configPath = "/home/pi/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Object_Detection_Files/frozen_inference_graph.pb"

# Initialize the DNN model for object detection using the config and weights files
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize the tracker
tracker = cv2.TrackerKCF_create()
trackers = cv2.MultiTracker_create()

# Function to count the number of people in the image
def detectPeople(img, thres, nms):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    people_boxes = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className == "person":
                people_boxes.append(box)
    return people_boxes

# Function to display information on the image
def displayInfo(img, boxes):
    for box in boxes:
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        cv2.putText(img, "Person", (int(box[0]) +10, int(box[1]) +30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

# Function to log the footfall count
def logFootfall(count):
    with open("/home/pi/Object_Detection_Files/footfall_log.txt", "a") as logFile:
        logFile.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, {count}\n")

if __name__ == "__main__":
    # Initialize Raspberry Pi camera
    cap = cv2.VideoCapture(1)  # Use device index 0 for Raspberry Pi camera module
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height

    footfall_count = 0
    start_time = time.time()

    while True:
        success, img = cap.read()
        if not success:
            break

        people_boxes = detectPeople(img, 0.5, 0.2)
        for box in people_boxes:
            trackers.add(cv2.TrackerCSRT_create(), img, tuple(box))
            footfall_count += len(people_boxes)
        
        # Update trackers and get the positions of tracked objects
        success, boxes = trackers.update(img)
        if success:
            displayInfo(img, boxes)

        cv2.putText(img, f"Footfall count: {footfall_count}", (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Output", img)

        if time.time() - start_time >= 3600:
            logFootfall(footfall_count)
            footfall_count = 0  # Reset footfall count after logging
            start_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
