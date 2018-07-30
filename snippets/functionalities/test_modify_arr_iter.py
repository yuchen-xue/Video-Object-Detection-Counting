import numpy as np

bbox1 = (580, 570, 100, 90)
bbox2 = (700, 590, 100, 80)
bbox3 = (770, 590, 90, 80)
boxes = np.stack((bbox1, bbox2, bbox3))
for cnt in range(10):
    if cnt % 2 == 0:
        for box in boxes:
            box[0] = box[0] + cnt * 3
            box[1] = box[1] + cnt * 3

    print(boxes)
