import imutils
import dlib
from imutils import face_utils

SHAPE_PREDICTOR_LOCATION = './shape_predictor.dat'

def resize_to(x=500):
    def resize(image):
        return imutils.resize(image, width=x)
    return resize

def get_face_rects(image):
    detector = dlib.get_frontal_face_detector()
    rects = detector(image, 1)

    return (image, list(rects))

def get_face_shape(image, rect):
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_LOCATION)
    shape = predictor(image, rect)
    shape = face_utils.shape_to_np(shape)

    return shape

def image_to_shape(image_rep):
    image = image_rep[0]
    rects = image_rep[1]
    shapes = [get_face_shape(image, rect) for rect in rects]
    return (image, shapes)

def preprocess(image_rep):
    image = image_rep[0]
    shape = image_rep[1][0]
    min_x, min_y, max_x, max_y = max_from_shape(shape)
    cropped = image[(min_y-10):(max_y+10), (min_x-10):(max_x+10)]
    resized = imutils.resize(cropped, height=200)

    return resized

def max_from_shape(shape):
    min_x = 9999
    min_y = 9999
    max_x = 0
    max_y = 0

    for x, y in shape:
        if x > max_x:
            max_x = x
        elif x < min_x:
            min_x = x
        if y > max_y:
            max_y = y
        elif y < min_y:
            min_y = y

    return min_x, min_y, max_x, max_y