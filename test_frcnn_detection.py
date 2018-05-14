"""Load live stream video from surveillance cam for a certain period and make predictions on it. """
import os
import time
import numpy as np
import cv2


VIDEO = 'data/test.avi'
PROTOTXT = 'models/MobileNetSSD_deploy.prototxt.txt'
MODEL = 'models/MobileNetSSD_deploy.caffemodel'


cap = cv2.VideoCapture(VIDEO)

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)


for _ in range(100):
    ret, frame = cap.read()
    blob = cv2.dnn.blobFromImage(frame, 1, (224, 224), (102.9801, 115.9465, 122.7717))
    
    net.setInput(blob)
    start = time.time()
    preds = net.forward()
    end = time.time()
    print("[INFO] classification took {:.5} seconds".format(end - start))

    print(" -- Predicts: -- \n"
          , preds)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
