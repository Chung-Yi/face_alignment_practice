import cv2

import dlib

import numpy

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

predictor = dlib.shape_predictor(PREDICTOR_PATH) # 利用predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")在捕捉的臉部預測臉部 landmarks

cascade_path='haarcascade_frontalface_default.xml'


cascade = cv2.CascadeClassifier(cascade_path)

def get_landmarks(im):

    rects = cascade.detectMultiScale(im, 1.3, 8) # 檢測出圖片中所有的人臉，並將人臉用vector保存各個人臉的座標、大小（用矩形表示）
    # print(rects)
    for (x,y,w,h) in rects:
    
        rect=dlib.rectangle(x,y,x+w,y+h)
        # print(rect)
        # count = 0
        # for p in predictor(im, rect).parts():
        #     print(p)
        #     count +=1
        # print(count)

        dimface = numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()]) # 找出68個點
        
        
        

        im = annotate_landmarks(im, dimface)

    return im

def annotate_landmarks(im, landmarks):

    im = im.copy()
    print(type(landmarks))

    for idx, point in enumerate(landmarks):
        # print(type(point))
        # print(point, point[0,0],point[0,1])
        pos = (point[0, 0], point[0, 1])

        cv2.putText(im, str(idx), pos,

                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,

                    fontScale=0.4,

                    color=(0, 0, 255))

        cv2.circle(im, pos, 4, color=(0, 255, 255))

    return im

im=cv2.imread('000010.jpg')

cv2.imwrite('output.jpg', get_landmarks(im))
