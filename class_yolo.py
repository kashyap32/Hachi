

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
from feature_extractor import FeatureExtractor

class YOLO_MODEL:
	def __init__(self):
		self.test=0

	def create_boxes(self,path_for_file,file_name):

		set_conf=0.5
		set_threshold=0.3
		# load the COCO class labels our YOLO model was trained on
		# labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
		labelsPath =  "yolo-coco/coco.names"
		LABELS = open(labelsPath).read().strip().split("\n")

		# initialize a list of colors to represent each possible class label
		np.random.seed(42)
		COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
			dtype="uint8")

		

		weightsPath="yolo-coco/yolov3.weights"
		configPath ="yolo-coco/yolov3.cfg"

		# load our YOLO object detector trained on COCO dataset (80 classes)
		print("[INFO] loading YOLO from disk...")
		net = cv2.dnn.readNet(configPath, weightsPath)
		#net.summary

		# load our input image and grab its spatial dimensions

		image = cv2.imread(path_for_file)
		orig_image=cv2.imread(path_for_file)
		# print(type(image))
		(H, W) = image.shape[:2]

		# determine only the *output* layer names that we need from YOLO
		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

		# construct a blob from the input image and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes and
		# associated probabilities
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()

		# show timing information on YOLO
		print("[INFO] YOLO took {:.6f} seconds".format(end - start))

		# initialize our lists of detected bounding boxes, confidences, and
		# class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []
		return_class=[]
		unique_class=[]
		maps=[]
		image_paths=[]

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability) of
				# the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

					# filter out weak predictions by ensuring the detected
					# probability is greater than the minimum probability
				if confidence > set_conf:
						# scale the bounding box coordinates back relative to the
						# size of the image, keeping in mind that YOLO actually
						# returns the center (x, y)-coordinates of the bounding
						# box followed by the boxes' width and height
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						# use the center (x, y)-coordinates to derive the top and
						# and left corner of the bounding box
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						# update our list of bounding box coordinates, confidences,
						# and class IDs
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)

		# apply non-maxima suppression to suppress weak, overlapping bounding
		# boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, set_conf,set_threshold)
		#print("This is count of boxes",len(boxes))

		#print(boxes)
		
		part_count=0
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				part_count=part_count+1
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				box_frame=image[y:y+h,x:x+w]
				# cv2.imshow('olay',box_frame)
				# cv2.waitKey(0)
				
				color = [int(c) for c in COLORS[classIDs[i]]]

				# print(x,y)
				# print(x+w,y+h)
				cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,0), 4) # Center by x+w/2,y+h/2
				temp=[x,y,x+w,y+h]
				crop_img=orig_image[y:y+h, x:x+w]
				temp_path="static/detected/"+file_name+"_part%d" % part_count
				cv2.imwrite(temp_path, crop_img)
				image_paths.append(temp_path)
				# cv2.imshow("anything",crop_img)
				# cv2.waitKey(0)
				# temp={'TL_x':x,'TL_y':y,'BR_x':x+w,'BR_y':y+h};
				maps.append(temp)
				# cv2.circle(image, (x+int(w/2),y+int(h/2)),2, (255,255,255), 3)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
				if (LABELS[classIDs[i]]) not in return_class:
					unique_class.append(LABELS[classIDs[i]])
				return_class.append(LABELS[classIDs[i]])
				# return_class.append(text)
				# cv2.putText(image, text, (x+int(w/2), y+int(h/2)), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

		# result = pred.img_process(image)

		# show the output image
		# cv2.imshow("Image", image)
		detected_img_path="static/detected/"+file_name
		cv2.imwrite(detected_img_path,image)
	
		return return_class,maps,image_paths,unique_class
