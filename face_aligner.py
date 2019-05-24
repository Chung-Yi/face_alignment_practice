from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import numpy as np
import imutils 
import dlib
import cv2
import os
import shutil

# 利用 detector = dlib.get_frontal_face_detector()預測圖像，評估有幾張臉在這張圖像中。
detector = dlib.get_frontal_face_detector() 
# 利用predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")在捕捉的臉部預測臉部 landmarks
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

fa = FaceAligner(predictor, desiredFaceWidth=200)

face_name = 1
def detect_face_landmarks(img):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 偵測人臉
    faces = detector(gray, 1)
    
    dir_path1 = 'gen'
    dir_path2 = 'ori'
    
    if not os.path.isdir(dir_path1):
        print('exist')
        os.mkdir(dir_path1)
    else:
        shutil.rmtree(dir_path1)
        os.mkdir(dir_path1)
    
    if not os.path.isdir(dir_path2):
        print('exist')
        os.mkdir(dir_path2)
    else:
        shutil.rmtree(dir_path2)
        os.mkdir(dir_path2)


    for face in faces:
        # 將dlib矩形轉換為OpenCV樣式的邊界框[即（x，y，w，h）]，然後繪製邊界框
        (x,y,w,h) = rect_to_bb(face)
        faceOrig = imutils.resize(img[y:y+h, x:x+w], width=200)
        # fa.align參數分別是要擷取的圖像、要被辨識的圖像(灰階)、要對齊的圖像
        faceAligned = fa.align(img, gray, face)

        global face_name
        cv2.imwrite('ori/faceOrig_{}.jpg'.format(face_name), faceOrig)
        cv2.imwrite('gen/faceAligned_{}.jpg'.format(face_name), faceAligned)
        face_name += 1


if __name__ == '__main__':
    image = 'human.jpg'
    detect_face_landmarks(image)