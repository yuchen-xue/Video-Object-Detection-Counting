"""Load live stream video from surveillance cam for a certain period and make predictions on it. """
import os
import time
import numpy as np
import cv2


VIDEO = os.environ['LIVE_STREAM']
PROTOTXT = os.environ['PROTOTXT']
MODEL = os.environ['MODEL']
LABELS = os.environ['SYNSET_WORDS']

rows = open(LABELS).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

cap = cv2.VideoCapture(VIDEO)

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)


for _ in range(100):
    ret, frame = cap.read()
    blob = cv2.dnn.blobFromImage(frame, 1, (224, 224), (104, 117, 123))
    
    net.setInput(blob)
    start = time.time()
    preds = net.forward()
    end = time.time()
    print("[INFO] classification took {:.5} seconds".format(end - start))
    
    # sort the indexes of the probabilities in descending order (higher
    # probabilitiy first) and grab the top-5 predictions
    preds = preds.reshape((1, len(classes)))
    idxs = np.argsort(preds[0])[::-1][:5]
    
    # loop over the top-5 predictions and display them
    for (i, idx) in enumerate(idxs):
        if i == 0:
            text = "Label: {}, {:.2f}%".format(classes[idx],
                                               preds[0][idx] * 100)
            cv2.putText(frame, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
                                                                classes[idx], preds[0][idx]))

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
