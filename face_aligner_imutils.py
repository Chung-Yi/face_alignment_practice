import cv2
import dlib
import numpy as np
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils


# 使用Dlib的dlib.get_frontal_face_detector偵測並取得相片中人物的臉部，評估有幾張臉在這張圖像中。
detector = dlib.get_frontal_face_detector()
# 利用predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")在捕捉的臉部預測臉部 landmarks
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)

image = cv2.imread('twice.jpg')
image = imutils.resize(image, width=1200)
# cv2.imwrite('ooo.jpg', image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('ooo_gray.jpg', gray)
# show the original input and detect faces in the grayscale
rects = detector(gray,2)

i=0

for rect in rects:
    # extract the ROI of the *original* face, then align the face
    # using facial landmarks
    # 將臉部區域交給imutils.face_utils的rect_to_bb取得x, y, w, h值，
    # 用以將臉部圖片縮放至指定的大小，最後送至FaceAligner輸出align的臉部圖片。
    (x,y,w,h) = rect_to_bb(rect)
    faceOrig = imutils.resize(image[y:y+h, x:x+w], width=256)
    # fa.align參數分別是要擷取的圖像、要被辨識的圖像(灰階)、要對齊的圖像
    faceAligned = fa.align(image, gray, rect)

    cv2.imwrite('ori-'+str(i)+'.jpg', faceOrig)
    cv2.imwrite('ali-'+str(i)+'.jpg', faceAligned)
    i+=1