from FP import Maybe, List
from os_helpers import get_directories, get_files_from_root
from feature_helpers import extract_image_features, process_image, split_data
from ml_helpers import experiment, train_model, save_model, load_model, predict_with_model
import numpy as np

# OPTIMIZATIONS TO BE DONE (hopefully)
# - Parallelize
# - Memoize or lru_cache
# - Fix constants
# - Iteration to recursion

IMAGE_DIR = './images'
EMOTIONS = ['angry', 'happy', 'neutral', 'sad']

def train(image_dir):
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

# Train and save the model
train(IMAGE_DIR)

# Predict emotion of image
# print(predict(['./happy-person.jpg', './happy-person.jpg']))

# Remove DS Store from images: rm ./app/images/*/.DS_Store