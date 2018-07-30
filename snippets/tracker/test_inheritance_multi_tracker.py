"""https://github.com/opencv/opencv_contrib/blob/master/modules/tracking/samples/multitracker.py"""
import itertools
import numpy as np
import cv2


class MyMultiTrackerCreate(cv2.MultiTracker):
    new_id = itertools.count

    def __init__(self):
        super().__init__()
        self.id = next(self.new_id())


cv2.namedWindow("tracking")
camera = cv2.VideoCapture('data/sample-video.avi')
init_once = False
tracker = None
boxes = tuple()

ok, frame = camera.read()
(fr_h, fr_w) = frame.shape[:2]
if not ok:
    print('Failed to read video')
    exit()

cnt = 0
while camera.isOpened():
    ok, frame = camera.read()
    if not ok:
        print('no frame to read')
        break

    if cnt % 2 == 0:
        # del tracker
        # concatenate a new box. Ref:
        # https://stackoverflow.com/a/22732845/8809592
        # if tracker is not None:
        # boxes = np.vstack([boxes, np.array([[770, 590, 90, 80]])]) if len(boxes) else boxes
        boxes = np.array([[770, 590, 90, 80], [520, 480, 55, 69]])
        for box in boxes:
            box[0] = box[0] - cnt * 2
            box[1] = box[1] - cnt * 2
        tracker = MyMultiTrackerCreate.create()
        print(tracker, tracker.id)
        init_once = False

    print(boxes)
    if not init_once:
        for i, box in enumerate(boxes):
            if not (0 < box[0] < fr_w) & (0 < box[1] < fr_h) & (box[0] + box[2] < fr_w) & (box[1] + box[3] < fr_h) \
                   & (box[2] != 0) & (box[3] != 0):
                boxes = np.delete(boxes, np.where(np.all(boxes == box, axis=1)), 0)
            else:
                ok = tracker.add(cv2.TrackerMIL_create(), frame, tuple(box))
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
