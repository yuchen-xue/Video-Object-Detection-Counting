"""Object detection and counting on videos"""
import time
import numpy as np
import cv2

VIDEO = 'data/sample-video.avi'
PROTO = 'models/MobileNetSSD_deploy.prototxt.txt'
MODEL = 'models/MobileNetSSD_deploy.caffemodel'
# initialize the list of class labels MobileNet SSD was trained to detect
OBJ_TYPES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat',
             'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
             'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
             'sofa', 'train', 'tvmonitor')
# bounding box colors for each class
BOX_COLOR = np.random.uniform(0, 255, size=(21, 3))
# counter color
COUNTER_COLOR = (0, 0, 255)


def initializer(video, proto, model):
    print('[INFO] loading model...')
    net = cv2.dnn.readNetFromCaffe(proto, model)

    print('[INFO] starting video stream...')
    cap = cv2.VideoCapture(video)
    time.sleep(1.0)
    return cap, net


def forward_net(frame_, net_):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame_, (300, 300)), 0.007843, (300, 300), 127.5)
    net_.setInput(blob)
    output = net_.forward()
    return output


def gen_obj_cnt(detections_):
    arr = detections_[0, 0, :, 1:3]
    arr = arr[np.all(arr != 0, axis=1)]
    sliced = arr[:, 0]
    unique, counts = np.unique(sliced, return_counts=True)
    stacked = np.column_stack((unique.astype('int'), counts))
    for obj in stacked:
        yield OBJ_TYPES[obj[0]], obj[1]


def gen_box_info(detections_, h_, w_):
    for i in np.arange(0, detections_.shape[2]):
        conf = detections_[0, 0, i, 2]
        idx = int(detections_[0, 0, i, 1])

        # filter out weak predictions.
        if conf > 0.0 and idx in (6, 7, 14, 15):
        # if conf > 0.0:
            box = detections_[0, 0, i, 3:7] * np.array([w_, h_, w_, h_])
            (startX, startY, endX, endY) = box.astype('int')
            obj_name = OBJ_TYPES[idx]

            yield obj_name, conf * 100, startX, startY, endX, endY


def main():
    cap, net = initializer(VIDEO, PROTO, MODEL)

    ok, frame = cap.read()
    if not ok:
        print('Failed to read video')
        exit()
    fr_h, fr_w = frame.shape[:2]

    # Frame counter
    cap_cnt = 1

    while cap.isOpened():
        _, frame = cap.read()

        # get prediction on current frame
        detections = forward_net(frame, net)

        # Display objects count
        for j, (obj, obj_cnt) in enumerate(gen_obj_cnt(detections)):
            cv2.putText(img=frame,
                        text='{}: {}'.format(obj, obj_cnt),
                        org=(5, 40 * (j + 1)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=COUNTER_COLOR,
                        thickness=2)

        # draw the prediction on the frame
        for obj_name, conf, startX, startY, endX, endY in gen_box_info(detections, fr_h, fr_w):
            label = '{}: {:.2f}%'.format(obj_name, conf)
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), BOX_COLOR[0], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR[0], 2)

        # show the output frame
        cv2.imshow('Campus Objects Detection', frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord('q'):
            break

        cap_cnt += 1

    # do a bit of cleanup
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
