"""Object detection and counting on videos"""
import time
import numpy as np
import cv2


def initializer(video, proto, model):
    print('[INFO] loading model...')
    net = cv2.dnn.readNetFromCaffe(proto, model)

    print('[INFO] starting video stream...')
    cap = cv2.VideoCapture(video)
    time.sleep(1.0)
    return cap, net


def foward_net(frame_, net_):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame_, (300, 300)), 0.007843, (300, 300), 127.5)
    net_.setInput(blob)
    return net_.forward()


def gen_box_info(frame_, detections_):
    # initialize the list of class labels MobileNet SSD was trained to detect
    cls_lst = ('background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
               'sofa', 'train', 'tvmonitor')
    (h, w) = frame_.shape[:2]

    for i in np.arange(0, detections_.shape[2]):
        conf = detections_[0, 0, i, 2]
        idx = int(detections_[0, 0, i, 1])

        # filter out weak predictions.
        # if conf > 0.0 and idx in (6, 7, 14, 15):
        if conf > 0.0:
            box = detections_[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            obj_name = cls_lst[idx]

            yield obj_name, conf * 100, startX, startY, endX, endY


def main():
    # bounding box colors for each class
    box_color = np.random.uniform(0, 255, size=(21, 3))
    # counter color
    cnt_color = (0, 0, 255)

    cap, net = initializer('data/sample-video.avi',
                           'models/MobileNetSSD_deploy.prototxt.txt',
                           'models/MobileNetSSD_deploy.caffemodel')

    # Frame counter
    cap_cnt = 1

    while cap.isOpened():
        _, frame = cap.read()

        # initiate an empty list for containing predicted objects
        obj_list = []

        # get prediction on current frame
        detections = foward_net(frame, net)
        print(cap_cnt, detections[0, 0, :, 1:3])

        # Process detection results
        box_info = gen_box_info(frame, detections)

        # draw the prediction on the frame
        for obj_name, conf, startX, startY, endX, endY in box_info:
            label = '{}: {:.2f}%'.format(obj_name, conf)
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), box_color[0], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color[0], 2)
            obj_list.append(obj_name)

        # Display objects count
        for j, (pred_class, n_class) in enumerate(x for x in set(map(lambda x: (x, obj_list.count(x)), obj_list))):
            cv2.putText(img=frame,
                        text='{}: {}'.format(pred_class, n_class),
                        org=(5, 40 * (j + 1)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=cnt_color,
                        thickness=2)
        
        # show the output frame
        cv2.imshow('Frame', frame)
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
