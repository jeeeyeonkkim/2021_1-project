import dlib # dlib -> face detection & recognition
import cv2 
import numpy as np # 행렬 연산

detector = dlib.get_frontal_face_detector() # 얼굴 탐지 모델
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat') # 얼굴 랜드마크 탐지 모델
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat') # 얼굴 인식 모델 

def find_faces(img): # 얼굴 찾는 함수 input: 이미지
    dets = detector(img, 1) # 얼굴 찾은 결과물

    if len(dets) == 0: # 얼굴 못 찾으면 빈 배열 리턴
        return np.empty(0), np.empty(0), np.empty(0)
    
    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int) # 얼굴의 특징점 68개 점 추출
    for k, d in enumerate(dets): # 얼굴마다 루프를 돈다 
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = sp(img, d) # 랜드마크 결과물
        
        # convert dlib shape to numpy array
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)
        
    return rects, shapes, shapes_np

def encode_faces(img, shapes): # 얼굴을 인코드 하는 함수 // enc..128개의 벡터를 구하는 과정
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)

# 먼저 기존의 np.load를 np_load_old에 저장해둠.
np_load_old = np.load
## 기존의 parameter을 바꿔줌
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

descs = np.load('img/descs.npy')[()]

def encode_face(img): # 인코드 이미지 
  dets = detector(img, 1)

  if len(dets) == 0:
    return np.empty(0)

  for k, d in enumerate(dets):
    shape = sp(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shape)

    return np.array(face_descriptor)

img_paths = { # 이미지 정보 불러오기 
    'neo': 'img/neo.jpg',
    'trinity': 'img/trinity.jpg',
    'morpheus': 'img/morpheus.jpg',
    'smith': 'img/smith.jpg',
    'jeeyeon': 'img/jyeon.jpg',
    'jeyoung': 'img/jyoung.jpg'
}

descs = { # 정보가 들어가기 위한 빈 dic
    'neo': None,
    'trinity': None,
    'morpheus': None,
    'smith': None,
    'jeeyeon': None,
    'jeyoung': None
}

for name, img_path in img_paths.items():
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    _, img_shapes, _ = find_faces(img_rgb)
    descs[name] = encode_faces(img_rgb, img_shapes)[0]

np.save('img/descs.npy', descs)
#print(descs) # 128개의 점을 출력하는 것...dic 

cap = cv2.VideoCapture(0) #웹캠에서 영상을 읽어온다.
cap.set(3, 640) #WIDTH
cap.set(4, 480) #HEIGHT

if not cap.isOpened():
  exit()

_, img_bgr = cap.read() # (800, 1920, 3)
padding_size = 0
resized_width = 1920
video_size = (resized_width, int(img_bgr.shape[0] * resized_width // img_bgr.shape[1]))
output_size = (resized_width, int(img_bgr.shape[0] * resized_width // img_bgr.shape[1] + padding_size * 2))

while True:
  ret, img_bgr = cap.read()
  if not ret:
    break

  img_bgr = cv2.resize(img_bgr, video_size)
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

  dets = detector(img_bgr, 1)

  for k, d in enumerate(dets):
    shape = sp(img_rgb, d)
    face_descriptor = facerec.compute_face_descriptor(img_rgb, shape)

    last_found = {'name': 'unknown', 'dist': 0.6, 'color': (0,0,255)}

    for name, saved_desc in descs.items():
      dist = np.linalg.norm([face_descriptor] - saved_desc, axis=1)

      if dist < last_found['dist']:
        last_found = {'name': name, 'dist': dist, 'color': (255,255,255)}

    cv2.rectangle(img_bgr, pt1=(d.left(), d.top()), pt2=(d.right(), d.bottom()), color=last_found['color'], thickness=2)
    cv2.putText(img_bgr, last_found['name'], org=(d.left(), d.top()), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=last_found['color'], thickness=2)

  #화면에 출력한다. 
  cv2.imshow('img', img_bgr)
  if cv2.waitKey(1) == ord('q'):
    break

cap.release()
#writer.release()