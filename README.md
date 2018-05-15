# LiveVideoObjectDetection
Load video to make object detection and counting on it.

Inspired by: 

https://www.pyimagesearch.com/2017/08/21/deep-learning-with-opencv/

Create two new directory, rename them as "models" and "data". 

download [MobileNetSSD_deploy.caffemodel](https://github.com/chuanqi305/MobileNet-SSD/blob/master/MobileNetSSD_deploy.caffemodel) and   

download [MobileNetSSD_deploy.prototxt](https://github.com/chuanqi305/MobileNet-SSD/blob/master/MobileNetSSD_deploy.prototxt)

put them into the "models" folder. 


Grab what ever video you like and put it into the "data" folder. 


In command prompt, type "python3 video_object_counter_caffemodel.py --video video_filename", then enjoy. 
