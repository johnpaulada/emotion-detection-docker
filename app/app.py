from FP import List
from os_helpers import get_directories, get_files_from_root
from feature_helpers import extract_image_features

IMAGE_DIR = './images'
SAMPLE_IMAGE = './images/3. sad/vh6to.jpg'

data = List(get_directories(IMAGE_DIR)) \
            .chain(get_files_from_root(IMAGE_DIR)) \
            .reduce(lambda v, acc: acc + v) \
            .map(extract_image_features) \
            .value

# processed = extract_image_features(SAMPLE_IMAGE, 0)

print(data)