import os
from os import listdir
from os.path import isfile, join


def all_files_in_path(path):
    fs = [join(path, f) for f in listdir(path)
          if isfile(join(path, f)) and
          f != ".DS_Store" and f != "base_processed.csv"]
    return fs
