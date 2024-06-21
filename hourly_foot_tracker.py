import cv2  # Import the OpenCV library
import time  # Import the time library
from datetime import datetime  # Import the datetime library

classNames = []  # Initialize an empty list for class names
classFile = "/home/pi/Object_Detection_Files/coco.names"  # Path to the coco.names file
with open(classFile, "rt") as f:  # Open the file in read mode
    classNames = f.read().rstrip("\n").split("\n")  # Read class names and split into a list

# Paths to the configuration and weights files for the SSD MobileNet model
configPath = "/home/pi/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"  # Path to the model config file
weightsPath = "/home/pi/Object_Detection_Files/frozen_inference_graph.pb"  # Path to the model weights file

net = cv2.dnn_DetectionModel(weightsPath, configPath)  # Load the DNN model
net.setInputSize(320, 320)  # Set the input size for the model
net.setInputScale(1.0 / 127.5)  # Set the input scale for the model
net.setInputMean((127.5, 127.5, 127.5))  # Set the mean values for input
net.setInputSwapRB(True)  # Swap the Red and Blue channels in the input image

# Initialize the tracker
tracker = cv2.TrackerKCF_create()  # Create a KCF tracker instance
trackers = cv2.MultiTracker_create()  # Create a MultiTracker instance

# Function to count the number of people in the image
def detectPeople(img, thres, nms):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)  # Detect objects in the image
    people_boxes = []  # Initialize an empty list for people bounding boxes
    if len(classIds) != 0:  # If any objects are detected
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):  # Iterate through detected objects
            className = classNames[classId - 1]  # Get the class name of the detected object
            if className == "person":  # Check if the detected object is a person
                people_boxes.append(box)  # Append the bounding box to people_boxes list
    return people_boxes  # Return the list of people bounding boxes

# Function to display information on the image
def displayInfo(img, boxes):
    for box in boxes:  # Iterate through the bounding boxes
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)  # Draw a rectangle around each detected person
        cv2.putText(img, "Person", (int(box[0]) + 10, int(box[1]) + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Put text "Person" near the rectangle

# Function to log the footfall count
def logFootfall(count):
    with open("/home/pi/Object_Detection_Files/footfall_log.txt", "a") as logFile:  # Open the log file in append mode
        logFile.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, {count}\n")  # Write the current timestamp and count

if __name__ == "__main__":
    # Initialize Raspberry Pi camera
    cap = cv2.VideoCapture(1)  # Use device index 1 for the Raspberry Pi camera module
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set the width of the video frame
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set the height of the video frame

    footfall_count = 0  # Initialize footfall count to zero
    start_time = time.time()  # Record the start time

    while True:
        success, img = cap.read()  # Read a frame from the video capture
        if not success:  # If frame reading is unsuccessful
            break  # Exit the loop

        people_boxes = detectPeople(img, 0.5, 0.2)  # Detect people in the image
        for box in people_boxes:  # Iterate through the detected people boxes
            trackers.add(cv2.TrackerCSRT_create(), img, tuple(box))  # Add the bounding box to the tracker
            footfall_count += len(people_boxes)  # Increment the footfall count
        
        # Update trackers and get the positions of tracked objects
        success, boxes = trackers.update(img)  # Update the trackers
        if success:  # If tracker update is successful
            displayInfo(img, boxes)  # Display the tracking information on the image

        cv2.putText(img, f"Footfall count: {footfall_count}", (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Display the footfall count on the image
        cv2.imshow("Output", img)  # Show the image in a window

        if time.time() - start_time >= 3600:  # Check if an hour has passed
            logFootfall(footfall_count)  # Log the footfall count
            footfall_count = 0  # Reset the footfall count
            start_time = time.time()  # Reset the start time

        if cv2.waitKey(1) & 0xFF == ord('q'):  # If 'q' key is pressed
            break  # Exit the loop

    cap.release()  # Release the video capture
    cv2.destroyAllWindows()  # Close all OpenCV windows
