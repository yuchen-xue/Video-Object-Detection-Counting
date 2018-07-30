"""https://github.com/opencv/opencv_contrib/blob/master/modules/tracking/samples/multitracker.py"""
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


def initializer(video, proto, model):
    cv2.namedWindow("tracking")
    print('[INFO] loading model...')
    net = cv2.dnn.readNetFromCaffe(proto, model)

    print('[INFO] starting video stream...')
    cap = cv2.VideoCapture(video)
    # time.sleep(1.0)
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


def get_trk_box(detections_, h_, w_):
    trk_list = list()
    for i in np.arange(0, detections_.shape[2]):
        conf = detections_[0, 0, i, 2]
        idx = int(detections_[0, 0, i, 1])

        # filter out weak predictions.
        if conf > 0.0 and idx in (6, 7, 14, 15):
        # if conf > 0.0:
            box = detections_[0, 0, i, 3:7] * np.array([w_, h_, w_, h_])
            (startX, startY, endX, endY) = box.astype('int')
            trk_width = endX - startX
            trk_height = endY - startY
            trk_list.append([startX, startY, trk_width, trk_height])

    # print(trk_list)
    if len(trk_list) is not 0:
        trk_box = np.vstack(tuple(trk_list))
        return trk_box
    else:
        return tuple()


def main():
    camera, net = initializer(VIDEO, PROTO, MODEL)

    init_once = False
    tracker = None
    boxes = tuple()

    ok, frame = camera.read()
    if not ok:
        print('Failed to read video')
        exit()
    fr_h, fr_w = frame.shape[:2]

    cnt = 0
    while camera.isOpened():
        ok, frame = camera.read()
        if not ok:
            print('no frame to read')
            break

        # get prediction on current frame
        detections = forward_net(frame, net)

        # Display objects count
        for j, (obj, obj_cnt) in enumerate(gen_obj_cnt(detections)):
            cv2.putText(img=frame,
                        text='{}: {}'.format(obj, obj_cnt),
                        org=(5, 40 * (j + 1)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=2)

        if cnt % 5 == 0:
            del tracker
            # concatenate a new box. Ref:
            # https://stackoverflow.com/a/22732845/8809592
            # boxes = np.vstack([boxes, np.array([[770, 590, 90, 80]])]) if len(boxes) else np.array(
            #     [[580, 570, 100, 90]])
            boxes = get_trk_box(detections, fr_h, fr_w)
            print(boxes)
            for box in boxes:
                box[0] = box[0] - cnt * 2
                box[1] = box[1] - cnt * 2
            tracker = cv2.MultiTracker_create()
            init_once = False

        if not init_once:
            for i, box in enumerate(boxes):
                if not (0 < box[0] < fr_w) & (0 < box[1] < fr_h) & (box[0] + box[2] < fr_w) & (box[1] + box[3] < fr_h) \
                       & (box[2] != 0) & (box[3] != 0):
                    boxes = np.delete(boxes, np.where(np.all(boxes == box, axis=1)), 0)
                else:
                    tracker.add(cv2.TrackerMIL_create(), frame, tuple(box))
            init_once = True

        ok, boxes = tracker.update(frame)

        for tkr_box in list(boxes):
            p1 = (int(tkr_box[0]), int(tkr_box[1]))
            p2 = (int(tkr_box[0] + tkr_box[2]), int(tkr_box[1] + tkr_box[3]))
            cv2.rectangle(frame, p1, p2, (200, 0, 0))

        cv2.imshow('tracking', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break  # esc pressed

        cnt += 1


if __name__ == '__main__':
    main()
