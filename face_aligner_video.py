from imutils.face_utils import FaceAligner
from imutils.video import count_frames
import imutils
import numpy as np
import dlib 
import cv2
import os
import shutil

# 利用 detector = dlib.get_frontal_face_detector()預測圖像，評估有幾張臉在這張圖像中。
detector = dlib.get_frontal_face_detector()
# 利用predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")在捕捉的臉部預測臉部 landmarks
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=64)

face_name = 1

def detect(img, idx, totle):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    print(totle)

    for face in faces:
        # fa.align參數分別是要擷取的圖像、要被辨識的圖像(灰階)、要對齊的圖像
        faceAligned = fa.align(gray, gray, face)
        global face_name
        cv2.imwrite('./face/{0}.jpg'.format(face_name),faceAligned)
        face_name += 1
        
    print('Working with {0} frames. completed {1:.2f}'.format(idx, idx/float(totle)*100))


detect_video = 'onionman.mp4'
videoCapture = cv2.VideoCapture(detect_video)
success, frame = videoCapture.read()
frame_counter = 1
frame_totle = count_frames(detect_video)

path = 'face'
if not os.path.isdir(path):
    os.mkdir(path)
else:
    shutil.rmtree(path)
    os.mkdir(path)

while success:
    
    detect(frame, frame_counter, frame_totle)
    success, frame = videoCapture.read()
    frame_counter += 1

print('Done!')