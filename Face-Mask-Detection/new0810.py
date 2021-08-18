# USAGE
# python detect_mask_video.py

# import the necessary packages
from typing import ByteString
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import random
import cv2
import os
import time
import socket

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

#수신에 사용될 내 ip와 내 port번호
TCP_IP = '192.168.0.76'
TCP_PORT = 5001
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(True)
conn, addr = s.accept()

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()

flag_x = 0	# motor 제어 변수
flag_y = 0
locs_index = 0
j = 0
d_x = 1
d_y = 1
# loop over the frames from the video stream
while True:
	length = recvall(conn,16) #길이 16의 데이터를 먼저 수신하는 것은 여기에 이미지의 길이를 먼저 받아서 이미지를 받을 때 편리하려고 하는 것이다.
	stringData = recvall(conn, int(length))
	data = np.fromstring(stringData, dtype='uint8')
	decimg=cv2.imdecode(data,1)
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = decimg
	frame = imutils.resize(frame, width=800)
	
	
	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
	loc_lenArray = []
	index_array = []
	# loop over the detected face locations and their corresponding
	# locations
	people = len(locs)
	if people > 9 : # 카메라 내 사람 수 제한
		people = 9

	if people == 0 :
		people_num = str(people)
		conn.send(people_num.encode('utf8')) # 카메라 내 사람이 없는 경우 send
		
	else :
		
		if j%15 == 0: # 15초 마다 focusing box 전환
			locs_focus = locs[random.randint(0, len(locs)-1)]

		locs_focus_len = locs_focus[2] - locs_focus[0] # boundary box x축 길이
		print(locs_focus_len)
		if locs_focus_len >= 100 and locs_focus_len <= 260: # 인식하는 bb 길이 제한
			people_num = str(people) 
			conn.send(people_num.encode('utf8')) # 카메라 내 사람 있는 경우 send

			sound = 0 # 파이보 소리 크기
			if people == 2:
				#2명일 경우
				sound = 50 
			elif people >=3 :
				#3명 이상일 경우
				sound = 80 
			else:
				if locs_focus_len >= 190: #가까울 경우
					sound = 50 
				else: #멀 경우
					sound = 60  # Set the volume to 60%.
			
			p_x = 6 
			p_y = 3 # PID 제어를 위한 P 값
			
			mid_x = (locs_focus[0] + locs_focus[2]) / 2
			mid_y = (locs_focus[1]  + locs_focus[3]) / 2
			print("x 중간점 : ", mid_x - 400, "y 중간점 : ", mid_y - 400)
			
			if mid_x - 400 > 170 : # eye tracking 알고리즘
				if flag_x < 40 :
					flag_x += p_x * d_x
				else :
					flag_x = -20
			elif mid_x - 400 > 160 :
				flag_x += 0.6
			elif mid_x - 400 < -190 :
				if flag_x > -40 :
					flag_x -= p_x * d_x
				else :
					flag_x = 20
			elif mid_x - 400 < -180:
				flag_x -= 0.6
			else :
				d_x = 1

			if mid_y - 400 > 55 :
				if flag_y < 25 :
					flag_y += p_y * d_y
				
			elif mid_y - 400 < -183 :
				if flag_y > -25 :
					flag_y -= p_y * d_y
				
			else :
				d_y = 1		
			# print ("p_x : ", p_x , " | p_y : ", p_y)
			# print("flag_x : ", flag_x)
			# print("flag_y : ", flag_y)		
			flags = []
			flags.append(flag_x)
			flags.append(flag_y)
			flags.append(sound)
			conn.send(str(flags).encode('utf8'))
			flags = []

		else :
			people = 0
			motion_send = str(people)
			conn.send(motion_send.encode('utf8'))
			
		j += 1
		if d_x > 0.3 : # PID 제어를 위한 d 값
			d_x -= 0.11
		else :
			d_x = 0
		
		if d_y > 0.1 :
			d_y -= 0.07
		else :
			d_y = 0
		

	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No qMask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			
		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
s.close()
# do a bit of cleanup
cv2.destroyAllWindows()
# vs.stop()