from FP import List
import os

def get_directories(path):
    return [directory for directory in os.listdir(path) if os.path.isdir(os.path.join(path, directory))]

def files_from_dir(directory, root):
    full_root_path = os.path.join(root, directory)
    files_in_directory = os.listdir(full_root_path)
    full_path_files_in_directory = [os.path.join(os.path.join(root, directory), f) for f in files_in_directory]

    return full_path_files_in_directory

def filename_with_index(index):
    def create_filename_tuple(filename, i):
        return (filename, index)

    return create_filename_tuple

def get_files_from_root(root):
    def get_files(directory, index):
        return List(files_from_dir(directory, root)) \
                    .map(filename_with_index(index))

    return get_files