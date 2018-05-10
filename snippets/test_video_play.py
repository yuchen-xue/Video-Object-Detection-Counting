import cv2

cap = cv2.VideoCapture('data/output.avi')


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# while cap.isOpened():
for _ in range(100):
    ret, frame = cap.read()
    cv2.putText(frame, 'text', (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
