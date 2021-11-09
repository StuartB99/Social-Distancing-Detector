# Social-Distancing-Detector
A simple Python system which detects and tracks humans and their distance from each other. The system uses the pre-trained YoloV3 object detection model to detect humans, which can be found here: (https://pjreddie.com/darknet/yolo/).

Annotates green and red bounding boxes on people who are following and not following social distance protocols respectively. Shows a counter of the total number of social distancing violations.

**Before Running:**
1. Install the pre-trained weight file yolov3.weights from https://pjreddie.com/media/files/yolov3.weights. This is required for the Yolov3 to run. This was not added to the Github project because Github has a file size limit of a 100 MB.
2. Move the file into the yolo-coco folder.

**To Run for Images:**
1. Open social_distancing_detector_image.py
2. Edit img_dir in line 10 to your image directory
3. Run


**To Run Video:**
1. Move video file to folder that the social_distancing_detector_video.py file is in (optional)
2. Parameters of social_distancing_detector_video.py file:
	- input : input video file name
	- ouput : output video, has to .avi file
	- display: 0 or 1, display=0 means no display when processing, display=1 means display frame by frame of the annotation process
3. Run command in the console: 
```
run social_distance_detector_video.py --input shopping_mall.mp4 --output output_shopping_mall.avi --display 1
```   

