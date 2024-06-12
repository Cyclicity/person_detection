import cv2  # Import the OpenCV library 

#thres = 0.45 # Threshold to detect object

classNames = []  # Initializing an empty list to store class names(strings)
classFile = "/home/pi/Object_Detection_Files/coco.names"  # File path for COCO names (string)
with open(classFile, "rt") as f:  # Open the class file in read text mode
    classNames = f.read().rstrip("\n").split("\n")  # Read and split class names by newline (list of strings)

configPath = "/home/pi/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"  # Path to the configuration file (string)
weightsPath = "/home/pi/Object_Detection_Files/frozen_inference_graph.pb"  # Path to the weights file (string)

net = cv2.dnn_DetectionModel(weightsPath, configPath)  # Load the detection model with config and weights 
net.setInputSize(320, 320)  # Set the input size for the model
net.setInputScale(1.0 / 127.5)  # Set the input scale for the model 
net.setInputMean((127.5, 127.5, 127.5))  # Set the input mean for the model 
net.setInputSwapRB(True)  # Swap the red and blue channels 

def getObjects(img, thres, nms, draw=True, objects=[]):
    """
    Detect objects in the image using the neural network model.
    
    img: numpy.ndarray - Input image in which to detect objects
    thres: float - Confidence threshold for detecting objects
    nms: float - Non-maximum suppression threshold
    draw: bool - Whether to draw bounding boxes and labels on the image
    objects: list - List of object class names to detect; if empty, detect all
    
    Returns:
    img: numpy.ndarray - Image with detected objects drawn
    objectInfo: list - List of detected objects with their bounding boxes and class names
    """
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)  # Detect objects (classIds: numpy.ndarray, confs: numpy.ndarray, bbox: list)
    
    if len(objects) == 0: 
        objects = classNames  # Use all class names if no specific objects are provided (list of strings)
    
    objectInfo = []  # Initialize an empty list to store detected object information (list)
    
    if len(classIds) != 0:  # If any objects are detected
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):  # Loop through each detected object's data
            className = classNames[classId - 1]  # Get the class name for the detected object (string)
            if className in objects:  # Check if the detected object's class name is in the objects list
                objectInfo.append([box, className])  # Append bounding box and class name to objectInfo 
                if draw:  # If drawing is enabled
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)  # Draw a rectangle around the detected object 
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), 
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Put the class name text above the rectangle 
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30), 
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Put the confidence text next to the class name 
    
    return img, objectInfo  # Return the modified image and the detected object information ((numpy.ndarray, list))

if __name__ == "__main__":  # If this script is run directly
    cap = cv2.VideoCapture(0)  # Open the default camera 
    cap.set(3, 640)  # Set the width of the video frame to 640 pixels 
    cap.set(4, 480)  # Set the height of the video frame to 480 pixels
    
    while True:  # Start an infinite loop to process video frames
        success, img = cap.read()  # Read a frame from the camera (success: bool, img: numpy.ndarray)
        result, objectInfo = getObjects(img, 0.45, 0.2)  # Detect objects in the frame (result: numpy.ndarray, objectInfo: list)
        cv2.imshow("Output", img)  # Display the frame with detected objects 
        cv2.waitKey(1)  # Wait for 1 millisecond before processing the next frame (int)
