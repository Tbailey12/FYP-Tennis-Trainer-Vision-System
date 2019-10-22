from pathlib import Path
import numpy as np

def get_project_root():
    ''' Returns the path to the root project dir'''
    return Path(__file__).parent.parent

def rgb_to_gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

