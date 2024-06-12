import cv2

classNames = []
classFile = "/home/pi/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/pi/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def countPeople(img, thres, nms):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    num_people = 0
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className == "person":
                num_people += 1
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, "Person", (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, f"Confidence: {round(confidence*100, 2)}%", (box[0] + 10, box[1] + 60),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Number of people: {num_people}", (20, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return img

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()
        result = countPeople(img, 0.45, 0.2)
        cv2.imshow("Output", img)
        cv2.waitKey(1)
