import cv2
from math import atan2, degrees, sqrt
from image_helpers import resize_to, get_face_rects, get_face_shape, image_to_shape, preprocess
from FP import List, Maybe

def extract_image_features(image_rep, i=0):
    image_path = image_rep[0]
    index = image_rep[1]
    print("Extracting features from image #{i}...".format(i=i+1))
    return (process_image(image_path), index)

def process_image(image_path, i=0):
    return Maybe.of(cv2.imread(image_path)) \
                .map(resize_to(500)) \
                .map(get_face_rects) \
                .map(image_to_shape) \
                .map(preprocess) \
                .map(get_face_rects) \
                .map(image_to_shape) \
                .map(lambda x: x[1][0]) \
                .map(get_features) \
                .value

def split_data(value, acc):
    x = value[0]
    y = value[1]
    x_acc = acc[0]
    y_acc = acc[1]
    x_acc.append(x)
    y_acc.append(y)

    return (x_acc, y_acc)

def noisy_identity(some_text):
    def identity(value):
        print(some_text)
        return value
    return identity

def get_features(face_shape):
    features = []
    coordinates = get_coordinates_from_shape(face_shape)
    face_lines = get_relevant_face_lines(coordinates)
    face_line_lengths = get_face_line_lengths(face_lines)
    face_line_angles = get_face_line_angles(face_lines)
    features = features + face_line_lengths
    features = features + face_line_angles

    return features

def get_coordinates_from_shape(shape):
    coordinates = [(x, y) for x, y in shape]

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

def get_face_line_lengths(face_lines):
    return [get_line_length(face_line[0], face_line[1]) for face_line in face_lines]

def get_face_line_angles(face_lines):
    return [get_line_angle(face_line[0], face_line[1]) for face_line in face_lines]

def pairs_from_ranges(ranges):
    pairs = []
    for r in ranges:
        pair = get_overlapping_pairs(r[0], r[1])
        pairs = pairs + pair

    return pairs

def get_overlapping_pairs(start, end):
    return [(x, x+1) for x in range(start, end)]

def lines_to_coords(lines, coordinates):
    return [line_indices_to_coords(line, coordinates) for line in lines]

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