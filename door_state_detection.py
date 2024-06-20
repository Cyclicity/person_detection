import cv2

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

# Define the class index for the door
DOOR_CLASS_INDEX = 71

# Function to detect the state of the door (open or closed)
def detectDoorState(img, thres, nms):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    door_state = "Unknown"

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className == "door":
                # You may need to implement additional logic to determine the state of the door
                # For simplicity, assuming the door is closed if detected
                door_state = "Closed"
                break
        else:
            door_state = "Open"
    else:
        door_state = "Open"

    return door_state

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            door_state = detectDoorState(img, 0.45, 0.2)
            cv2.putText(img, f"Door state: {door_state}", (20, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Output", img)

            # Exit the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
