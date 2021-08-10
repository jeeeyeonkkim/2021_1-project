# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import socket
import threading 
from queue import Queue


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
TCP_PORT = 1222
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(True)
conn, addr = s.accept()

# initialize the video stream and allow the camera sensor to warm up

ADDR = ('172.18.139.50', 1111)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind(ADDR)  # 주소 바인딩
    server_socket.listen()  # 클라이언트의 요청을 받을 준비

SIZE = 1024

# loop over the frames from the video stream
while True:

	conn, addr = s.accept()  # 수신대기, 접속한 클라이언트 정보 (소켓, 주소) 반환
	msg = conn.recv(SIZE)  # 클라이언트가 보낸 메시지 반환
	print("[{}] message : {}".format(addr,msg))  # 클라이언트가 보낸 메시지 출력

	conn.sendall("welcome!".encode())  # 클라이언트에게 응답

	length = recvall(conn,16) #길이 16의 데이터를 먼저 수신하는 것은 여기에 이미지의 길이를 먼저 받아서 이미지를 받을 때 편리하려고 하는 것이다.
	stringData = recvall(conn, int(length))
	data = np.fromstring(stringData, dtype='uint8')
	decimg=cv2.imdecode(data,1)

	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = decimg
	frame = imutils.resize(frame, width=800)
	
	# show the output frame

	key = cv2.waitKey(1) & 0xFF

	conn.close()  # 클라이언트 소켓 종료	

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
