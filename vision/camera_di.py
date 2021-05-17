#!/usr/bin/python
import socket
import cv2
import numpy
import os
import sys
import time

# 상위 디렉토리 추가 (for utils.config)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.config import Config as cfg

# openpibo 라이브러리 경로 추가
sys.path.append(cfg.OPENPIBO_PATH + '/lib')
from motion.motionlib import cMotion

# openpibo 라이브러리 경로 추가
sys.path.append(cfg.OPENPIBO_PATH + '/lib')
from vision.visionlib import cCamera

#연결할 서버(수신단)의 ip주소와 port번호
TCP_IP = '172.18.152.170'
TCP_PORT = 1111
#송신을 위한 socket 준비
sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))

#이미지 읽어오기
#cap.set(3, 640) #WIDTH
#cap.set(4, 480) #HEIGHT

#얼굴 인식 캐스케이드 파일 읽기
face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')

#OpenCV를 이용해서 webcam으로 부터 이미지 추출
#cam = cCamera()
capture = cv2.VideoCapture(0)
capture.set(3, 640) #WIDTH
capture.set(4, 480) #HEIGHT

m = cMotion(conf=cfg)

m.set_motors(positions=[0,0,-70,-25,0,0,0,0,70,25], movetime=5000)
pastValue = 0
nowValue = 0
flag = 1
while True:
    #frame = cam.read()
    ret, frame = capture.read()
    print("here1..")
    #frame을 바꾸자..
    frame = cv2.flip(frame,0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    
    #인식된 얼굴 갯수를 출력
    print(len(faces))
    print(faces)
    
    #인식된 얼굴에 사각형을 출력
    for (x,y,w,h) in faces:
       
        print("변화량 : ")
        print(pastValue-nowValue)
        
        if x < 150 :
            m.set_motors(positions=[0,0,-70,-25,flag,0,0,0,70,25], movetime=1000)
            flag -= 4
        elif x > 350 :
            m.set_motors(positions=[0,0,-70,-25,flag,0,0,0,70,25], movetime=1000)
            flag += 4

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        pastValue = nowValue
        

    #추출한 이미지를 String 형태로 변환(인코딩)시키는 과정
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = numpy.array(imgencode)
    stringData = data.tostring()

    #String 형태로 변환한 이미지를 socket을 통해서 전송
    sock.send( str(len(stringData)).ljust(16).encode())
    sock.send( stringData )
    #print("11")

cv2.destroyAllWindows() 
sock.close()


