import numpy as np
import matplotlib
import cv2
import os

## my imports
import utils

def load_image(image_num, color):
    ''' loads an image from a directory full of images and returns cv2 mat'''
    img_list = os.listdir()
    return cv2.imread(img_list[image_num], color)

if __name__ == '__main__':
    print('localization.py')

    # ---- define directories ---- #
    img_input_dir = 'img\input'
    img_output_dir = 'img\output'

    # -- change to img directory -- #
    root_dir = utils.get_project_root()
    os.chdir(root_dir)

    os.chdir(img_input_dir)
    # -- load image -- #
    img_list = os.listdir()
    img = load_image(13,1)
    os.chdir(root_dir)

    # -- preprocessing -- #



    # -- show image -- #
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()