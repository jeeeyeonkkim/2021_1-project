# -*- coding: utf8 -*-
import cv2
import socket
import numpy as np
import time
import pickle
import os
import sys

# 상위 디렉토리 추가 (for utils.config)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.config import Config as cfg

# openpibo 라이브러리 경로 추가
sys.path.append(cfg.OPENPIBO_PATH + '/lib')
from motion.motionlib import cMotion

m = cMotion(conf=cfg)

global past_value
global derivative
global integration

'''
#IP = '192.168.120.61'
IP = '172.17.194.9' # HGU_WLAN_FREE
PORT = 5001

print("camera.py 시작")

## TCP 사용
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

## server ip, port
s.connect((IP, PORT))
print("소캣연결 완료")
'''

def socket_conneting(IP,PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    s.connect((IP, PORT))
    print("소캣연결 완료")

    return s

def move(n, degree, speed, accel):
  m.set_speed(n, speed)
  m.set_acceleration(n, accel)
  m.set_motor(n, degree)

def pidvalue(value):
    global past_value
    global derivative
    global integration
    p_value = 0.24
    i_value = 0.05
    d_value = 0.83

    derivative = value - past_value
    pid_value = float(p_value * value) + float(i_value * value) + float(d_value * derivative)
    past_value = value

    integration += derivative

    return pid_value



def tracking(s) :
    goal_position_x = 0
    goal_position_y = 0
    pre_motor_degree_x = 0
    pre_motor_degree_y = 0
    size = 980

    move(4,0,100,50)
    move(5,0,100,50)

    while True:
        
        data = s.recv(1024)
        data = pickle.loads(data)
        dx = data[0]
        dy = data[1]
        size = data[2]
        print(dx,dy,size)
        print("트래킹 중")

        if dx == -1 and dy == -1 and size == -1:
            move(4,0,100,50)
            move(5,0,100,50)
        else:
            if size < 10000000 and dx < -80:
                #dx 이동 4번 모터
                goal_position_x = pre_motor_degree_x + 6
                #move(4, goal_position_x, 100, 50)
                move(4, pre_motor_degree_x+1, 100, 50)
                pre_motor_degree_x += 0.5
                move(4, pre_motor_degree_x+1, 80, 50)
                pre_motor_degree_x += 1
                move(4, pre_motor_degree_x+1, 60, 50)
                pre_motor_degree_x += 1.5
                move(4, pre_motor_degree_x+1, 50, 30)
                pre_motor_degree_x += 1.5
                move(4, pre_motor_degree_x+1, 40, 30)
                pre_motor_degree_x += 1
                move(4, pre_motor_degree_x+1, 30, 30)
                pre_motor_degree_x += 0.5
                pre_motor_degree_x = goal_position_x
                

            elif size < 10000000 and dx > 80:
                #dx 이동 4번 모터
                goal_position_x = pre_motor_degree_x - 6
                #move(4, goal_position_x, 100, 50)
                move(4, pre_motor_degree_x-1, 100, 50)
                pre_motor_degree_x -= 0.5
                move(4, pre_motor_degree_x-1, 80, 50)
                pre_motor_degree_x -= 1
                move(4, pre_motor_degree_x-1, 60, 50)
                pre_motor_degree_x -= 1.5
                move(4, pre_motor_degree_x-1, 50, 30)
                pre_motor_degree_x -= 1.5
                move(4, pre_motor_degree_x-1, 40, 30)
                pre_motor_degree_x -= 1
                move(4, pre_motor_degree_x-1, 30, 30)
                pre_motor_degree_x -= 0.5
                pre_motor_degree_x = goal_position_x

            if size < 10000000 and dy < -80:
                #dy 이동 5번 모터
                goal_position_y = pre_motor_degree_y + 6
                #move(5, goal_position_y, 100, 50)
                move(5, pre_motor_degree_y+1, 100, 50)
                pre_motor_degree_y += 0.5
                move(5, pre_motor_degree_y+1, 80, 50)
                pre_motor_degree_y += 1
                move(5, pre_motor_degree_y+1, 60, 50)
                pre_motor_degree_y += 1.5
                move(5, pre_motor_degree_y+1, 50, 30)
                pre_motor_degree_y += 1.5
                move(5, pre_motor_degree_y+1, 40, 30)
                pre_motor_degree_y += 1
                move(5, pre_motor_degree_y+1, 30, 30)
                pre_motor_degree_y += 0.5
                pre_motor_degree_y = goal_position_y

            elif size < 10000000 and dy > 80:
                #dy 이동 5번 모터
                goal_position_y = pre_motor_degree_y - 6
                #move(5, goal_position_y, 100, 50)
                move(5, pre_motor_degree_y-1, 100, 50)
                pre_motor_degree_y -= 0.5
                move(5, pre_motor_degree_y-1, 80, 50)
                pre_motor_degree_y -= 1
                move(5, pre_motor_degree_y-1, 60, 50)
                pre_motor_degree_y -= 1.5
                move(5, pre_motor_degree_y-1, 50, 30)
                pre_motor_degree_y -= 1.5
                move(5, pre_motor_degree_y-1, 40, 30)
                pre_motor_degree_y -= 1
                move(5, pre_motor_degree_y-1, 30, 30)
                pre_motor_degree_y -= 0.5
                pre_motor_degree_y = goal_position_y

        #time.sleep(0.5)

'''
while True:

    

    
    # 비디오의 한 프레임씩 읽는다.
    # 제대로 읽으면 ret = True, 실패면 ret = False, frame에는 읽은 프레임
    time_elapsed = time.time() - prev
    ret1, frame = cam.read()
    if time_elapsed > 1./frame_rate:
        prev = time.time()

        #frame = cv2.flip(frame,0) # 사진 상하 반전
        frame = cv2.flip(frame,0) # 사진 좌우 반전
        string = time.ctime()
        x1, y1 = 30, 30
        cv2.putText(frame, text=string, org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        # cv2. imencode(ext, img [, params])
        # encode_param의 형식으로 frame을 jpg로 이미지를 인코딩한다.
        ret2, frame = cv2.imencode('.jpg', frame, encode_param)
        # frame을 String 형태로 변환
        data = np.array(frame)
        stringData = data.tostring()
    
        #서버에 데이터 전송
        #(str(len(stringData))).encode().ljust(16)
        #time.sleep(1)
        #s.sendall((str(len(stringData))).encode().ljust(16) + stringData)
        s.send((str(len(stringData))).encode().ljust(16) + stringData)
        data = s.recv(1024)
        #print(data.decode())
    

    cam.release()
'''

if __name__ == "__main__":

    global past_value
    global derivative
    global integration
    past_value = 0
    derivative = 0
    integration = 0
    #socket = socket_conneting('172.17.194.9',5007)
    socket = socket_conneting('172.18.141.45',5007)
    print("프로그램 시작") 
    tracking(socket)
    print("프로그램 종료")
