import cv2
import sys

# Read video
video = cv2.VideoCapture('data/sample-video.avi')

# Read first frame.
ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()

cnt = 0

# Define an initial bounding box
bbox = (580, 570, 100, 90)
box_2 = (700, 590, 100, 80)
tracker = cv2.TrackerBoosting_create()
tracker_2 = cv2.TrackerBoosting_create()

while video.isOpened():
    tracker = cv2.TrackerBoosting_create()
    if cnt % 10 == 0:
        bbox = (bbox[0] + 3 * cnt, bbox[1] + 3 * cnt, 100, 90)
    # Read a new frame
    tracker.init(frame, bbox)
    tracker_2.init(frame, box_2)
    ok, frame = video.read()

    if not ok:
        print("Not ok")
        break

    # Update tracker
    ok, bbox = tracker.update(frame)
    ok2, box_2 = tracker_2.update(frame)

    # Draw bounding box
    if ok & ok2:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        q1 = (int(box_2[0]), int(box_2[1]))
        q2 = (int(box_2[0] + box_2[2]), int(box_2[1] + box_2[3]))
        cv2.rectangle(frame, q1, q2, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

    cnt += 1
