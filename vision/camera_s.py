#!/usr/bin/python
import socket
import cv2
import numpy
import os
import sys

# 상위 디렉토리 추가 (for utils.config)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.config import Config as cfg

# openpibo 라이브러리 경로 추가
sys.path.append(cfg.OPENPIBO_PATH + '/lib')
from vision.visionlib import cCamera

#연결할 서버(수신단)의 ip주소와 port번호
TCP_IP = '192.168.0.76'
TCP_PORT = 1234
#송신을 위한 socket 준비
sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))


while True:

    #OpenCV를 이용해서 webcam으로 부터 이미지 추출
    cam = cCamera()
    frame = cam.read()
    cam.imwrite("test.jpg", frame)

    #추출한 이미지를 String 형태로 변환(인코딩)시키는 과정
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = numpy.array(imgencode)
    stringData = data.tostring()

    #String 형태로 변환한 이미지를 socket을 통해서 전송
    sock.send( str(len(stringData)).ljust(16).encode())
    sock.send( stringData )
    sock.close()

    #다시 이미지로 디코딩해서 화면에 출력. 그리고 종료
    decimg=cv2.imdecode(data,1)
    cv2.imshow('CLIENT',decimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


