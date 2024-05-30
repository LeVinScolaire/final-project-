#just to clarify, the other two files were donwloaded and do most of the heavy lifting
import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

#these are things that the program can recognize
CLASSES = [
    "background", "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorcycle", "person", "potted plant", "sheep", "sofa", "train", "monitor"
]

#targets the device's camera
cap = cv2.VideoCapture(0)

#big main loop that runs all of the stuff
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (300, 300))
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

#this takes what the ai thinks its seeing and assigns a name to it
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            if class_id < len(CLASSES):
                class_label = CLASSES[class_id]
                #this puts the box around what it's detecting 
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                #this draws the box and puts the previously detected name on it
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, class_label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #this shows what the camera is seeing along with all of the stuff the program does
    cv2.imshow('Camera View', frame)
    #lets you quit the program by pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
