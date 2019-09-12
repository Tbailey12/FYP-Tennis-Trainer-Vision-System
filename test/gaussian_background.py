import numpy as np

from glob import glob  # for batch loading files

import imageio  # loading and handling images

import PIL # image lib used for image manipulation

import matplotlib.pyplot as plt
import matplotlib.colors as colours

import time

import os
import utils

if __name__ == "__main__":
    root = utils.get_project_root()

    dir = "test"
    # img_dir = "img\\img_campus"
    img_dir = "img\\input"
    save_dir = "img\\output"

    scale = 0.5

    os.chdir(root)
    os.chdir(img_dir)

    # ---- Load and convert image colour ---- #
    # img_list = glob('trees****.bmp')  # create a list of all test images
    img_list = glob('****.png')
    img_list.sort()

    img = imageio.imread(img_list[0])  # read the first image in the list
    img_g = utils.rgb_to_gray(img)  # convert image to 8 bit gray
    img_g = np.array(PIL.Image.fromarray(img_g).resize((int((scale*img_g.shape[1])),int(scale*img_g.shape[0]))))

    img_mean = np.zeros(img_g.shape)
    img_std = np.zeros(img_g.shape)

    p = 0.1 # learning rate
    B = np.zeros(img_g.shape)
    A = np.zeros(img_g.shape)

    plt.figure(figsize=(15,np.ceil(15*img_g.shape[0]/(img_g.shape[1]*2))))

    time_count = 0

    for i,img_iter in enumerate(img_list):
        img_temp = utils.rgb_to_gray(imageio.imread(img_iter))  # convert image to 8 bit gray
        img_temp = np.array(PIL.Image.fromarray(img_temp).resize((int((scale*img_temp.shape[1])),int(scale*img_temp.shape[0]))))

        if i==0:    # set mean to first image
            img_mean = img_temp
            continue

        start = time.process_time()
        img_mean = (1 - p) * img_mean + p * img_temp  # calculate mean
        img_std = np.sqrt((1 - p) * (img_std ** 2) + p * ((img_temp - img_mean) ** 2))  # calculate std deviation

        B_old = B
        B = np.logical_or((img_temp > (img_mean + 2*img_std)),
                          (img_temp < (img_mean - 2*img_std)))  # foreground new
        A = ~np.logical_and(B_old, B)  # difference between prev foreground and new foreground
        C = np.logical_and(A, B)   # different from previous frame and part of new frame
        time_count+=time.process_time()-start


        # ---- Plotting and saving ---- #
        plt.subplot(1,2,1)
        plt.imshow(img_temp,cmap='gray')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(C.astype(int), cmap='gray')
        plt.axis('off')
        plt.tight_layout()

        os.chdir(root)
        os.chdir(save_dir)
        # plt.savefig(img_iter, format='png')
        os.chdir(root)
        os.chdir(img_dir)
        plt.pause(0.001)
        plt.clf()

    print(time_count/len(img_list))