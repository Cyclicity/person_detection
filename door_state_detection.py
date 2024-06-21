import cv2  # Import the OpenCV library

classNames = []  # Initialize an empty list for class names
classFile = "/home/pi/Object_Detection_Files/coco.names"  # Path to the coco.names file
with open(classFile, "rt") as f:  # Open the file in read mode
    classNames = f.read().rstrip("\n").split("\n")  # Read class names and split into a list

configPath = "/home/pi/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"  # Path to the model config file
weightsPath = "/home/pi/Object_Detection_Files/frozen_inference_graph.pb"  # Path to the model weights file

net = cv2.dnn_DetectionModel(weightsPath, configPath)  # Load the DNN model
net.setInputSize(320, 320)  # Set the input size for the model
net.setInputScale(1.0 / 127.5)  # Set the input scale for the model
net.setInputMean((127.5, 127.5, 127.5))  # Set the mean values for input
net.setInputSwapRB(True)  # Swap the Red and Blue channels in the input image

DOOR_CLASS_INDEX = 71  # The index for the "door" class in the model

# Function to detect the state of the door (open or closed)
def detectDoorState(img, thres, nms):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)  # Detect objects in the image
    door_state = "Unknown"  # Initialize door state as unknown

    if len(classIds) != 0:  # If any objects are detected
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):  # Iterate through detected objects
            className = classNames[classId - 1]  # Get the class name of the detected object
            if className == "door":  # Check if the detected object is a door
                door_state = "Closed"  # Set door state to closed
                break  # Exit the loop
        else:
            door_state = "Open"  # If no door is detected, set door state to open
    else:  # If no objects are detected
        door_state = "Open"  # Set door state to open

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

            # Exit the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' key is pressed
                break  # Exit the loop
    finally:
        cap.release()  # Release the video capture
        cv2.destroyAllWindows()  # Close all OpenCV windows
