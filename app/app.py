from FP import Maybe, List
from os_helpers import get_directories, get_files_from_root
from feature_helpers import extract_image_features, process_image, split_data
from ml_helpers import experiment, train_model, save_model, load_model, predict_with_model
import numpy as np
import argparse

# OPTIMIZATIONS TO BE DONE (hopefully)
# - Parallelize
# - Use splat
# - Memoize or lru_cache
# - Fix constants
# - Iteration to recursion

IMAGE_DIR = './images'
EMOTIONS = ['angry', 'happy', 'neutral', 'sad']

def train(image_dir=IMAGE_DIR):
    print("Training start...")
    
    List(get_directories(image_dir)) \
        .chain(get_files_from_root(image_dir)) \
        .reduce(lambda v, acc: acc + v) \
        .map(extract_image_features) \
        .reduce(split_data, ([], [])) \
        .map(lambda v: (np.array(v[0]), np.array(v[1]))) \
        .map(experiment) \
        .map(train_model) \
        .map(save_model)

    print("Training complete.")

def predict(image_paths):
    print("Predicting...")

    return List(image_paths) \
        .map(process_image) \
        .reduce(lambda v, acc: acc + (v,), ()) \
        .map(predict_with_model(load_model())) \
        .map(lambda v: [EMOTIONS[x] for x in v]) \
        .value

parser = argparse.ArgumentParser(description='Emotion Detection Application')
parser.add_argument('--train', nargs=1, help='train the program to predict emotions', metavar='path_to_training_folder')
parser.add_argument('--predict', nargs='+', help='predict the emotion of face in image', metavar='path_to_input_image')
args = parser.parse_args()
unpacked_args = vars(args)
train_arg = unpacked_args['train']
predict_arg = unpacked_args['predict']

if (train_arg is not None):
    train(train_arg[0])

if (predict_arg is not None):
    print(predict(predict_arg))

# Remove DS Store from images: rm ./app/images/*/.DS_Store