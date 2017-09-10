import cv2
import dlib
from imutils import face_utils

IMAGE_PATH = 'rheena.jpg'
DOTTED_IMAGE_PATH = 'rheena-dots.jpg'
detector = dlib.get_frontal_face_detector()
image = cv2.imread(IMAGE_PATH)
rects = detector(image, 1)
predictor = dlib.shape_predictor('./shape_predictor.dat')
for rect in rects:
    shape = predictor(image, rect)
    shape = face_utils.shape_to_np(shape)
    
    for i, (x, y) in enumerate(shape):
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(image, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)

cv2.imwrite(DOTTED_IMAGE_PATH, image)