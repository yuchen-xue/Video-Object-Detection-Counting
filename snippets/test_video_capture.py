import os
import cv2


cap = cv2.VideoCapture(os.environ['LIVE_STREAM'])

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('../data/output.avi', fourcc, 20.0, (1920, 1080))
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

finally:
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
