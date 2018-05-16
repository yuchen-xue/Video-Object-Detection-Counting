"""Object detection and counting on videos"""
import time
import argparse
import numpy as np
import cv2
import imutils
from imutils.video import FPS

# initialize arguments settings
ap = argparse.ArgumentParser()
ap.add_argument("-V", "--video", type=str,
                help="path to video you wanna play on")
ap.add_argument("-P", "--prototxt", default='models/MobileNetSSD_deploy.prototxt.txt',
                help="path to Caffe prototxt file")
ap.add_argument("-M", "--model", default='models/MobileNetSSD_deploy.caffemodel',
                help="path to Caffe pre-trained model")
ap.add_argument("-W", "--width", type=int, default=0, 
                help="width of the video display")
ap.add_argument("-H", "--height", type=int, default=0, 
                help="height of the video display")
ap.add_argument("-R", "--reshape", type=bool, default=False, 
                help="Whether reshape the video. Default is False")
ap.add_argument("-C", "--confidence", type=float, default=0.2,
                help='minimum probability to filter weak detections')
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
           'sofa', 'train', 'tvmonitor']
# bounding box colors for each class
BBOX_COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# counter color
COUNTER_COLOR = (0, 0, 255)


def main():
    print('[INFO] loading model...')
    net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

    print('[INFO] starting video stream...')
    cap = cv2.VideoCapture(args['video'])
    time.sleep(1.0)
    fps = FPS().start()

    # loop over the frames from the video stream
    while cap.isOpened():
        _, frame = cap.read()
        if args['reshape'] is True:
            frame = imutils.resize(frame, width=args['width'], height=args['height'])

        # grab the frame dimensions, convert it to a blob and feed the network
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # initiate an empty list for containing predicted objects
        obj_list = []

        # loop over detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # filter out weak predictions.
            if confidence > args['confidence']:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                # draw the prediction on the frame
                label = '{}: {:.2f}%'.format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY),
                              (endX, endY), BBOX_COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BBOX_COLORS[idx], 2)
                obj_list.append(CLASSES[idx])

        # Display objects count
        for j, (pred_class, n_class) in enumerate(x for x in set(map(lambda x: (x, obj_list.count(x)), obj_list))):
            cv2.putText(img=frame,
                        text='{}: {}'.format(pred_class, n_class),
                        org=(5, 40 * (j + 1)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=COUNTER_COLOR,
                        thickness=2)

        # show the output frame
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord('q'):
            break

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print('[INFO] elapsed time: {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
