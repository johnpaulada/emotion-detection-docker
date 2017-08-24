import numpy as np
import imutils
import dlib
import cv2
import os
from math import atan2, degrees, sqrt
from imutils import face_utils
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib

# OPTIMIZATIONS
# - PCA
# - List comprehensions
# - Move related functions to other modules
# - Iteration to recursion
# - Memoize or lru_cache
# - Make declarative

def train(x, y):
    ert_classifier = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    ert_classifier.fit(x, y)

    return ert_classifier

def process_photos(images_path):
    directories = [directory for directory in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, directory))]
    data_x = []
    data_y = []
    
    for index, directory in enumerate(directories):
        directory = os.path.join(images_path, directory)
        for image in os.listdir(directory):
            image_path = os.path.join(directory, image)
            print(image_path)
            photo_features = process_photo(image_path)
            data_x.append(photo_features)
            data_y.append(index)

    data_x_array = np.array(data_x)
    data_y_array = np.array(data_y)

    return data_x_array, data_y_array

def process_photo(image_path):
    image = cv2.imread(image_path)

    image = imutils.resize(image, width=500)
    face_rects = get_face_rects(image)

    for rect in face_rects:
        shape = get_face_shape(image, rect)
        processed_image = preprocess(image, shape)
        processed_face_rects = get_face_rects(processed_image)
        for processed_face_rect in processed_face_rects:
            print('==========')
            processed_shape = get_face_shape(processed_image, processed_face_rect)
            features = get_features(processed_shape)

            return features

def get_face_rects(image):
    detector = dlib.get_frontal_face_detector()
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(image, 1)

    return rects

def get_face_shape(image, rect):
    predictor = dlib.shape_predictor('./shape_predictor.dat')
    shape = predictor(image, rect)
    shape = face_utils.shape_to_np(shape)

    return shape

def preprocess(image, shape):
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

    cropped = image[(min_y-10):(max_y+10), (min_x-10):(max_x+10)]
    resized = imutils.resize(cropped, height=200)

    return resized

def get_features(face_shape):
    features = []
    coordinates = get_coordinates_from_shape(face_shape)
    face_lines = get_relevant_face_lines(coordinates)
    face_line_lengths = get_face_line_lengths(face_lines)
    face_line_angles = get_face_line_angles(face_lines)
    features = features + face_line_lengths
    features = features + face_line_angles

    return features

def get_face_line_lengths(face_lines):
    line_lengths = []

    for face_line in face_lines:
        point1 = face_line[0]
        point2 = face_line[1]
        line_length = get_line_length(point1, point2)
        line_lengths.append(line_length)

    return line_lengths

def get_face_line_angles(face_lines):
    line_angles = []

    for face_line in face_lines:
        point1 = face_line[0]
        point2 = face_line[1]
        line_angle = get_line_angle(point1, point2)
        line_angles.append(line_angle)

    return line_angles

def get_coordinates_from_shape(shape):
    coordinates = []
    for x, y in shape:
        coordinates.append((x, y))

    return coordinates

def get_relevant_face_lines(coordinates):
    face_lines = []

    eye_ranges = [(36, 41), (42, 47)]
    brow_ranges = [(17, 21), (22, 26)]
    mouth_ranges = [(48, 59), (60, 67)]

    eye_indices = pairs_from_ranges(eye_ranges)
    brow_indices = pairs_from_ranges(brow_ranges)
    mouth_indices = pairs_from_ranges(mouth_ranges)

    eye_coords = lines_to_coords(eye_indices, coordinates)
    brow_coords = lines_to_coords(brow_indices, coordinates)
    mouth_coords = lines_to_coords(mouth_indices, coordinates)

    face_lines = face_lines + eye_coords
    face_lines = face_lines + brow_coords
    face_lines = face_lines + mouth_coords

    return face_lines

def pairs_from_ranges(ranges):
    pairs = []
    for r in ranges:
        pair = get_overlapping_pairs(r[0], r[1])
        pairs = pairs + pair

    return pairs

def lines_to_coords(lines, coordinates):
    line_coords = []
    for line in lines:
        line_coord = line_indices_to_coords(line, coordinates)
        line_coords.append(line_coord)
    
    return line_coords

def line_indices_to_coords(line_indices, coordinates):
    point1_index = line_indices[0]
    point2_index = line_indices[1]
    point1_coordinates = coordinates[point1_index]
    point2_coordinates = coordinates[point2_index]
    line_coordinates = (point1_coordinates, point2_coordinates)

    return line_coordinates

def get_line_length(point1, point2):
    delta_x, delta_y = get_points_delta(point1, point2)
    delta_x_squared = delta_x ** 2
    delta_y_squared = delta_y ** 2
    delta_squared_sum = delta_x_squared + delta_y_squared
    delta_squared_sum_root = sqrt(delta_squared_sum)
    length = delta_squared_sum_root

    return length

def get_line_angle(point1, point2):
    delta_x, delta_y = get_points_delta(point1, point2)
    angle_in_radians = atan2(delta_y, delta_x)
    angle_in_degrees = degrees(angle_in_radians)

    return angle_in_degrees

def get_points_delta(point1, point2):
    X_INDEX = 0
    Y_INDEX = 1
    delta_x = point2[X_INDEX] - point1[X_INDEX]
    delta_y = point2[Y_INDEX] - point1[Y_INDEX]

    return delta_x, delta_y

def draw_points(coordinates, image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (i, (x, y)) in enumerate(coordinates):
        cv2.putText(image, str(i), (x, y), font, 0.3, (0,255,0), 1, cv2.LINE_AA)

def get_overlapping_pairs(start, end):
    pairs = []
    for x in range(start, end):
        pairs.append((x, x+1))

    return pairs

# x, y = process_photos('./images')
# model = train(x, y)
# joblib.dump(model, 'emotion_detector.pkl')
emotions = ['angry', 'happy', 'neutral', 'sad']
model = joblib.load('emotion_detector.pkl')
test_features = process_photo('./aa.jpg')
raw_results = model.predict([test_features])
results = [emotions[x] for x in raw_results]
print(results)

# Draw rect on face
# x, y, w, h = face_utils.rect_to_bb(rect)
# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)