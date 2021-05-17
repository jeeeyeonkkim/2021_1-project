import os
import sys
import time
# 상위 디렉토리 추가 (for utils.config)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.config import Config as cfg

# openpibo 라이브러리 경로 추가
sys.path.append(cfg.OPENPIBO_PATH + '/lib')
from motion.motionlib import cMotion

if __name__ == "__main__":
  m = cMotion(conf=cfg)
  
  m.set_motion(name="m1", cycle=1) #up
  # m.set_motion(name="m2", cycle=1) #front
  #m.set_motion(name="m3", cycle=1) #down

  
  

