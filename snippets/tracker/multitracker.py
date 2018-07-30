"""https://github.com/opencv/opencv_contrib/blob/master/modules/tracking/samples/multitracker.py"""
import cv2

cv2.namedWindow("tracking")
camera = cv2.VideoCapture('data/sample-video.avi')
tracker = cv2.MultiTracker_create()
init_once = False

ok, image = camera.read()
if not ok:
    print('Failed to read video')
    exit()

bbox1 = (580, 570, 100, 90)
bbox2 = (700, 590, 100, 80)
bbox3 = (770, 590, 90, 80)

while camera.isOpened():
    ok, image = camera.read()
    if not ok:
        print('no image to read')
        break

    if not init_once:
        ok = tracker.add(cv2.TrackerMIL_create(), image, bbox1)
        ok = tracker.add(cv2.TrackerMIL_create(), image, bbox2)
        ok = tracker.add(cv2.TrackerMIL_create(), image, bbox3)
        init_once = True

    ok, boxes = tracker.update(image)
    print(ok, type(boxes))

    for newbox in list(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(image, p1, p2, (200, 0, 0))

    cv2.imshow('tracking', image)
    k = cv2.waitKey(1)
    if k == 27:
        break  # esc pressed
