import cv2
import time

# Load class names from coco.names file into a list
classNames = []
classFile = "/home/pi/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Find the index of "door"
try:
    door_index = classNames.index("door")  # Get the index of "door" from class names
    DOOR_CLASS_INDEX = door_index  # Assign the index to DOOR_CLASS_INDEX
except ValueError:
    print("The class 'door' is not found in coco.names.")  # Print error if "door" is not found
    exit()  # Exit the program if "door" is not found

# Paths to the configuration and weights files for the SSD MobileNet model
configPath = "/home/pi/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Object_Detection_Files/frozen_inference_graph.pb"

# Initialize the DNN model for object detection using the config and weights files
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)  # Set the input size for the model
net.setInputScale(1.0 / 127.5)  # Set the input scale for the model
net.setInputMean((127.5, 127.5, 127.5))  # Set the mean values for input
net.setInputSwapRB(True)  # Swap the Red and Blue channels in the input image

# Variables to track door state and time
door_state = "Unknown"  # Initialize door state as unknown
door_open_time = None  # Initialize door open time as None
alert_threshold = 120  # 2 minutes in seconds for the alert threshold

# Function to detect the state of the door (open or closed)
def detectDoorState(img, thres, nms):
    global door_state, door_open_time  # Use global variables door_state and door_open_time

    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)  # Detect objects in the image
    
    if len(classIds) != 0:  # If any objects are detected
        door_detected = False  # Initialize door_detected as False
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):  # Iterate through detected objects
            if classId == DOOR_CLASS_INDEX:  # Check if the detected object is a door
                door_state = "Closed"  # Set door state to closed
                door_detected = True  # Set door_detected to True
                door_open_time = None  # Reset the open time when door is detected as closed
                break  # Exit the loop
        if not door_detected:  # If no door is detected
            door_state = "Open"  # Set door state to open
            if door_open_time is None:  # If door_open_time is not set
                door_open_time = time.time()  # Set door_open_time to current time
            if time.time() - door_open_time >= alert_threshold:  # Check if alert threshold is reached
                print(f"Alert: Door has been open for {alert_threshold // 60} minutes. Please close the door.")  # Print alert message
                door_open_time = None  # Reset door open time to prevent repeated alerts
    else:  # If no objects are detected
        door_state = "Open"  # Set door state to open
        if door_open_time is None:  # If door_open_time is not set
            door_open_time = time.time()  # Set door_open_time to current time
        if time.time() - door_open_time >= alert_threshold:  # Check if alert threshold is reached
            print(f"Alert: Door has been open for {alert_threshold // 60} minutes. Please close the door.")  # Print alert message
            door_open_time = None  # Reset door open time to prevent repeated alerts

    return door_state  # Return the current state of the door

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Initialize video capture with the default camera
    cap.set(3, 640)  # Set the width of the video frame
    cap.set(4, 480)  # Set the height of the video frame

    try:
        while True:
            success, img = cap.read()  # Read a frame from the video capture
            if not success:  # If frame reading is unsuccessful
                break  # Exit the loop

            door_state = detectDoorState(img, 0.45, 0.2)  # Detect the state of the door
            cv2.putText(img, f"Door state: {door_state}", (20, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Display door state on the frame
            cv2.imshow("Output", img)  # Show the frame in a window

            if cv2.waitKey(1) & 0xFF == ord('q'):  # If 'q' key is pressed
                break  # Exit the loop
    finally:
        cap.release()  # Release the video capture
        cv2.destroyAllWindows()  # Close all OpenCV windows
