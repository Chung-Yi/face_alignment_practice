from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

"""
用 Haar的方式，在條件非常寬鬆的條件下，只有辨識到一張臉，甚至左下角也誤將商標誤認為臉部。
相較之下，Dlib較能正確得辨識出圖中的兩張臉

"""

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 利用 detector = dlib.get_frontal_face_detector()預測圖像，評估有幾張臉在這張圖像中。
detector = dlib.get_frontal_face_detector() 
# 利用predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")在捕捉的臉部預測臉部 landmarks
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def detect_haar(img):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 檢測出圖片中所有的人臉，並將人臉用vector保存各個人臉的座標、大小（用矩形表示）
    # [[225  75  89  89]]
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=3) 

    for (x,y,w,h) in faces:
        img = cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 3)
    cv2.imwrite('new.jpg', img)

def detect_face_landmarks(img):
    img  = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detector 函數的第二個參數是指定反取樣（unsample）的次數，如果圖片太小的時候，將其設為 1 可讓程式偵較容易測出更多的人臉。
    #偵測人臉
    faces = detector(gray,1)
    print(type(faces))

    count = 0
    for face in faces:
        # 確定面部區域的面部標誌，然後將面部標誌（x，y）座標轉換成NumPy陣列(68,2)
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        # 將dlib矩形轉換為OpenCV樣式的邊界框即[（x，y，w，h）]，然後繪製邊界框
        (x,y,w,h) = face_utils.rect_to_bb(face)
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 3)
        
        count += 1
        for idx, (x,y) in enumerate(shape):
            

            cv2.putText(img, str(idx), (x,y),

                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,

                    fontScale=0.1,

                    color=(0, 0, 255))
            cv2.circle(img, (x,y), 1, (0,0,255), -1)
        
    print(count)
    cv2.imwrite('landmarks.jpg', img)

if __name__ == '__main__':
    image = 'human.jpg'
    detect_haar(image)
    detect_face_landmarks(image)