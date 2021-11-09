from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
from imutils import paths
import numpy as np
import os
import imutils
import cv2

img_dir = "D:/static images"

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# loop over the image paths
for imagePath in paths.list_images(img_dir):
    image = cv2.imread(imagePath)   # read the image
    
    # resize the frame and then detect people (and only people) in it
    image = imutils.resize(image, width=700)
    orig = image.copy()             # copy the image for display comparison later on
    results = detect_people(image, net, ln, personIdx=LABELS.index("person"))
    
    # initialize the set of indexes that violate the minimum social distance
    violate = set()
    
     # ensure there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
    if len(results) >= 2:
		# extract all centroids from the results and compute the
		# Euclidean distances between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")
        print(centroids)
        print(D)
		# loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the configured number
				# of pixels
                if D[i, j] < config.MIN_DISTANCE:
					# update our violation set with the indexes of
					# the centroid pairs
                    violate.add(i)
                    violate.add(j)
    
    # loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
		# extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)
		# if the index pair exists within the violation set, then
		# update the color
        if i in violate:
            color = (0, 0, 255)
		# draw (1) a bounding box around the person and (2) the
		# centroid coordinates of the person,
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        cv2.circle(image, (cX, cY), 5, color, 1)
	# draw the total number of social distancing violations on the
	# output frame
    text = "Social Distancing Violations: {}".format(len(violate))
    cv2.putText(image, text, (10, image.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
    
    # show the output images
    cv2.imshow("Before Detections", orig)
    cv2.imshow("After Detections", image)
    k = cv2.waitKey(0)   
    
    if k == ord('q'): #quit when "q" is pressed
        break
    
cv2.destroyAllWindows() 
