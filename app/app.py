from FP import List
from os_helpers import get_directories, get_files_from_root
from feature_helpers import extract_image_features, process_image, split_data
from ml_helpers import experiment, train_model, save_model, load_model
import numpy as np

# OPTIMIZATIONS
# - PCA
# - List comprehensions
# - Iteration to recursion
# - Memoize or lru_cache
# - Try numpy on everything
# - Fix constants

# SAMPLE_IMAGE = './images/3. sad/vh6to.jpg'
IMAGE_DIR = './images'
EMOTIONS = ['angry', 'happy', 'neutral', 'sad']

# TODO: Make FP friendly
def train(image_dir):
    print("Getting data...")
    data = List(get_directories(image_dir)) \
            .chain(get_files_from_root(image_dir)) \
            .reduce(lambda v, acc: acc + v) \
            .map(extract_image_features) \
            .reduce(split_data, ([], [])) \
            .value
    
    x = np.array(data[0])
    y = np.array(data[1])

    print("Training...")
    experiment(x, y)

    print("Training...")
    model = train_model(x, y)

    print("Saving model...")
    save_model(model)

    print("Training complete.")

# TODO: Make FP friendly
def predict(image_path):
    model = load_model()
    data = process_image(image_path)
    raw_results = model.predict([data])
    results = [EMOTIONS[x] for x in raw_results]

    return results

# rm ./app/images/*/.DS_Store
# train(IMAGE_DIR)
# print(predict('./happy-person.jpg'))