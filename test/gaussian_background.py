import numpy as np

from glob import glob  # for batch loading files

import imageio  # loading and handling images

import matplotlib.pyplot as plt

import os
import utils

if __name__ == "__main__":
    root = utils.get_project_root()

    dir = "test"
    img_dir = "img\\img_campus"

    os.chdir(root)
    os.chdir(img_dir)

    # ---- Load and convert image colour ---- #
    img_list = glob('trees****.bmp')  # create a list of all test images
    img_list.sort()

    img = imageio.imread(img_list[0])  # read the first image in the list
    img_g = utils.rgb_to_gray(img).astype('uint8')  # convert image to 8 bit gray

    img_mean = np.zeros(img_g.shape)
    img_std = np.zeros(img_g.shape)

    p = 0.1  # learning rate
    B = np.zeros(img_g.shape)
    A = np.zeros(img_g.shape)
    for img_iter in img_list:
        img_temp = utils.rgb_to_gray(imageio.imread(img_iter)).astype('uint8')  # convert image to 8 bit gray

        img_mean = (1 - p) * img_mean + p * img_temp  # calculate mean
        img_std = np.sqrt((1 - p) * img_std ** 2 + p * (img_temp - img_mean) ** 2) # calculate std deviation

        A = ~np.logical_and(A,B)
        B = np.logical_or((img_temp>(img_mean+2*img_std)), (img_temp<(img_mean-2*img_std))) # foreground
        C = np.logical_and(A,B)

        plt.imshow(C.astype(int), cmap='gray')
        # plt.show()
        plt.pause(0.001)
        plt.clf()



    # for i in range(10):
    #     img = imageio.imread(img_list[i])                   # read the first image in the list
    #     img_g = utils.rgb_to_gray(img).astype('uint8')      # convert image to 8 bit gray
    #
    #     if i == 0:
    #         img_mean = img_g
    #
    #     img_mean += img_g

    # ---- show image ---- #
    plt.imshow(img_mean, cmap='gray')
    plt.axis('off')
    plt.show()
