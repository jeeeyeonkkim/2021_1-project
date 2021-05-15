import os
import sys

# 상위 디렉토리 추가 (for utils.config)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.config import Config as cfg

# openpibo 라이브러리 경로 추가
sys.path.append(cfg.OPENPIBO_PATH + '/lib')
from motion.motionlib import cPyMotion

import time

m = cPyMotion()
i = 9

def move(n, speed, accel, degree):
  m.set_speed(n, speed)
  m.set_acceleration(n, accel)
  m.set_motor(n, degree)

def test():
    
    move(i, 50, 0, 30)
    time.sleep(2)
  
    move(i, 50, 10, -30)
    time.sleep(2)

print("Init")
move(i, 20, 0, 0)
time.sleep(1)

print("Start")
test()

move(i, 20, 0, 0)
