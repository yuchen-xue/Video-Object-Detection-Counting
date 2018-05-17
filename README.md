# LiveVideoObjectDetection
Load video to make object detection and counting on it.

Inspired by: 

https://www.pyimagesearch.com/2017/08/21/deep-learning-with-opencv/

Install dependancies: 
```
pip install --upgrade opencv-python
pip install --upgrade imutils
```

1. Create two new directory, rename them as "models" and "data". 

2.download [MobileNetSSD_deploy.caffemodel](https://github.com/chuanqi305/MobileNet-SSD/blob/master/MobileNetSSD_deploy.caffemodel) and    

3. download [MobileNetSSD_deploy.prototxt](https://github.com/chuanqi305/MobileNet-SSD/blob/master/MobileNetSSD_deploy.prototxt) and rename the file as "MobileNetSSD_deploy.prototxt.txt"

4. put them into the "models" folder. 

5. Grab what ever video you like and put it into the "data" folder. 

6. In command prompt, type "python3 video_object_counter_caffemodel.py --video video_filename", then enjoy. 
