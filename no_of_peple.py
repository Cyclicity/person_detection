import cv2  # Import the OpenCV library
classNames = []  # List to store class names
classFile = "/home/pi/Object_Detection_Files/coco.names"  # Path to coco.names file
with open(classFile, "rt") as f:  # Open coco.names file in read text mode
    classNames = f.read().rstrip("\n").split("\n")  # Read file, strip trailing newline, and split by newline

configPath = "/home/pi/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"  # Path to config file
weightsPath = "/home/pi/Object_Detection_Files/frozen_inference_graph.pb"  # Path to weights file

net = cv2.dnn_DetectionModel(weightsPath, configPath)  # Create DNN model using config and weights
net.setInputSize(320, 320)  # Set input size for the model 
net.setInputScale(1.0 / 127.5)  # Set input scale for normalization
net.setInputMean((127.5, 127.5, 127.5))  # Set mean values for input normalization 
net.setInputSwapRB(True)  # Swap Red and Blue channels 

def countPeople(img, thres, nms):
    """
    Detect people in the image and count them.
    
    Parameters:
    img (numpy.ndarray): The input image in which to detect objects.
    thres (float): The confidence threshold for detection.
    nms (float): The non-maximum suppression threshold.
    
    Returns:
    img (numpy.ndarray): The image with detected objects annotated.
    """
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)  # Detect objects in the image
    num_people = 0  # Initialize the number of detected people to 0

    if len(classIds) != 0:  # If any objects are detected
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):  # Iterate through detected objects
            className = classNames[classId - 1]  # Get the class name for the detected object
            if className == "person":  # If the detected object is a person
                num_people += 1  # Increment the people counter
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)  # Draw a rectangle around the detected person
                cv2.putText(img, "Person", (box[0] + 10, box[1] + 30),  # Add the image with the class name
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, f"Confidence: {round(confidence*100, 2)}%", (box[0] + 10, box[1] + 60),  # Add with confidence
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Number of people: {num_people}", (20, 50),  # Display the number of people detected on the image
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return img  # Return the image

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Capture video from the camera
    cap.set(3, 640)  # Set the width of the video frame to 640 pixels
    cap.set(4, 480)  # Set the height of the video frame to 480 pixels

    while True:  # Loop to continuously capture frames from the camera
        success, img = cap.read()  # Read a frame from the camera
        result = countPeople(img, 0.45, 0.2)  # Detect people in the frame
        cv2.imshow("Output", img)  # Display the frame in a window
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop if the 'q' key is pressed
            break

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close any OpenCV windows
