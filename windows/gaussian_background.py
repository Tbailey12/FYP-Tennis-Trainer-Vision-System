import numpy as np

from glob import glob  # for batch loading files

import imageio  # loading and handling images
import cv2 as cv

import PIL # image lib used for image manipulation

import matplotlib.pyplot as plt
import matplotlib.colors as colours

from skimage import measure
from scipy import ndimage

import time

import os
import utils

class Candidate():
    def __init__(self, label):
        self.label = label
        self.size = None
        self.x = None
        self.y = None
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

if __name__ == "__main__":
    root = utils.get_project_root()

    dir = "test"
    # img_dir = "img\\img_campus"
    img_dir = "img/input_small"
    save_dir = "img/output"

    scale = 0.5

    os.chdir(root)
    os.chdir(img_dir)

    # ---- Load and convert image colour ---- #
    # img_list = glob('trees****.bmp')  # create a list of all test images
    img_list = glob('****.png')
    img_list.sort()

    img = cv.imread(img_list[0])  # read the first image in the list
    img_g = utils.rgb_to_gray(img)  # convert image to 8 bit gray
    img_g = img_g.astype(np.float32)
    # img_g = np.array(PIL.Image.fromarray(img_g).resize((int((scale*img_g.shape[1])),int(scale*img_g.shape[0]))),dtype=np.float32)

    img_mean = np.zeros(img_g.shape, dtype=np.float32)
    img_std = np.zeros(img_g.shape, dtype=np.float32)

    p = 0.1 # learning rate
    B = np.zeros(img_g.shape)
    A = np.zeros(img_g.shape)
    C = np.zeros(img_g.shape)

    mean_1 = np.zeros(img_g.shape)
    mean_2 = np.zeros(img_g.shape)

    std_1 = np.zeros(img_g.shape)
    std_2 = np.zeros(img_g.shape)
    std_3 = np.zeros(img_g.shape)
    std_4 = np.zeros(img_g.shape)
    std_5 = np.zeros(img_g.shape)
    std_6 = np.zeros(img_g.shape)

    B_1_std = np.zeros(img_g.shape)
    B_1_mean = np.zeros(img_g.shape)
    B_greater = np.zeros(img_g.shape)
    B_2_mean = np.zeros(img_g.shape)
    B_less = np.zeros(img_g.shape)

    # ---- Find objects ---- #
    # Filter params
    min_size = 2
    max_size = 400
    max_ratio = 2.5  # height to width ratio

    # plt.figure(figsize=(15,np.ceil(15*img_g.shape[0]/(img_g.shape[1]*2))))

    time_count = 0

    for i,img_iter in enumerate(img_list):
        img_temp = utils.rgb_to_gray(imageio.imread(img_iter))  # convert image to 8 bit gray
        img_temp = img_temp.astype(np.float32)
        # img_temp = np.array(PIL.Image.fromarray(img_temp).resize((int((scale*img_temp.shape[1])),int(scale*img_temp.shape[0]))),dtype=np.float32)

        if i==0:    # set mean to first image
            img_mean = img_temp
            continue

        start = time.time_ns()
        mid = time.time_ns()

        # img_mean = (1 - p) * img_mean + p * img_temp  # calculate mean

        # print(f"mean: {(time.time_ns()-mid)/1E6}")
        # mid = time.time_ns()

        ####################################################################################
        ############################## Background Subtraction ##############################

        np.multiply(1-p,img_mean,out=mean_1)
        np.multiply(p,img_temp,out=mean_2)
        np.add(mean_1,mean_2,out=img_mean)


        # print(f"mean2: {(time.time_ns()-mid)/1E6}")
        mid = time.time_ns()


        # img_std = np.sqrt((1 - p) * np.square(img_std) + p * np.square(img_temp - img_mean))  # calculate std deviation

        # print(f"std: {(time.time_ns()-mid)/1E6}")
        # mid = time.time_ns()

        np.square(img_std,out=std_1)
        np.multiply(1-p,std_1,out=std_2)
        np.subtract(img_temp,img_mean,out=std_3)
        np.square(std_3,out=std_4)
        np.multiply(p,std_4,out=std_5)
        np.add(std_2,std_5,out=std_6)
        np.sqrt(std_6,out=img_std)

        # print(f"std2: {(time.time_ns()-mid)/1E6}")
        mid = time.time_ns()

        B_old = B

        # print(f"B_old: {(time.time_ns()-mid)/1E6}")
        mid = time.time_ns()

        # B = np.logical_or((img_temp > (img_mean + 2*img_std)),
        #                   (img_temp < (img_mean - 2*img_std)))  # foreground new

        np.multiply(img_std,2,out=B_1_std)
        np.add(B_1_std,img_mean,out=B_1_mean)
        B_greater = np.greater(img_temp,B_1_mean)
        np.subtract(img_mean,B_1_std,out=B_2_mean)
        B_less = np.less(img_temp,B_2_mean)
        B = np.logical_or(B_greater,B_less)

        # print(f"B2: {(time.time_ns()-mid)/1E6}")
        mid = time.time_ns()

        A = ~np.logical_and(B_old, B)  # difference between prev foreground and new foreground

        # print(f"A: {(time.time_ns()-mid)/1E6}")
        mid = time.time_ns()

        C = np.logical_and(A, B)   # different from previous frame and part of new frame

        # print(f"C: {(time.time_ns()-mid)/1E6}")
        mid = time.time_ns()

        time_count+=time.process_time()-start

        # print((time.time_ns()-start)/1E6)

        ####################################################################################
        ################################# Object Detection #################################
        if i>0:
            start2 = time.time_ns()
            mid = time.time_ns()

            labels, n_features = ndimage.label(C)  # label image

            print(f"label: {(time.time_ns()-mid)/1E6}")
            mid = time.time_ns()

            C = 255*C.astype(np.uint8)
            n_features_cv, labels_cv, stats_cv, centroids_cv = cv.connectedComponentsWithStats(C, connectivity=8)

            label_mask_cv = np.logical_and(stats_cv[:,cv.CC_STAT_AREA]>1, stats_cv[:,cv.CC_STAT_AREA]<100)
            ball_candidates = np.concatenate((stats_cv[label_mask_cv],centroids_cv[label_mask_cv]), axis=1)
            # print(ball_candidates)

            # print(f"label_cv: {(time.time_ns()-mid)/1E6}")
            # mid = time.time_ns()

            # label_sizes = ndimage.sum(C, labels, index=range(n_features + 1))

            # print(f"sum: {(time.time_ns()-mid)/1E6}")
            # mid = time.time_ns()

            # # # mask all objects under a certain size
            # mask = np.bitwise_and(min_size < label_sizes, label_sizes < max_size)
            # labels_masked = mask[labels.ravel()].reshape(labels.shape)
            # labels[~labels_masked] = 0
            # candidates = []
            
            # labels_masked = 255*labels_masked.astype(np.uint8)
            # cv.imshow('img',C)
            # cv.waitKey(25)

            # print(f"mask: {(time.time_ns()-mid)/1E6}")
            # mid = time.time_ns()

            # # filtered_labels = [c.label for c in candidates]
            # objects = ndimage.find_objects(labels)

            # print(f"find_obj: {(time.time_ns()-mid)/1E6}")
            # mid = time.time_ns()

            # for j, obj in enumerate(objects):
            #     if obj is not None:
            #         label = j + 1
            #         c = Candidate(label)
            #         c.size = label_sizes[label]
            #         center = [int(s) for s in ndimage.center_of_mass(labels, labels, label)]
            #         c.x = center[1]
            #         c.y = center[0]
            #         c.xmin = obj[1].start - 1
            #         if c.xmin < 0:  c.xmin = 0
            #         c.xmax = obj[1].stop
            #         c.ymin = obj[0].start - 1
            #         if c.ymin < 0:  c.ymin = 0
            #         c.ymax = obj[0].stop
            #         # discard if proportions are outside bounds
            #         width = c.xmax - c.xmin
            #         height = c.ymax - c.ymin
            #         if (width / height > max_ratio) | (height / width < 1 / max_ratio):
            #             continue
            #         candidates.append(c)

            # print(f"candidates: {(time.time_ns()-mid)/1E6}")
            # mid = time.time_ns()

            print((time.time_ns()-start2)/1E6)


    #     plt.imshow(C, cmap='gray')
    #     plt.show()


        # if i>30:
        #     C = C.astype(np.uint8)
        #     plt.imshow(C, cmap='gray')
        #     plt.show()



        # ---- Plotting and saving ---- #
        # plt.subplot(1,2,1)
        # plt.imshow(img_temp,cmap='gray')
        # plt.axis('off')
        # plt.subplot(1,2,2)
        # plt.imshow(C.astype(int), cmap='gray')
        # plt.axis('off')
        # plt.tight_layout()

        # os.chdir(root)
        # os.chdir(save_dir)
        # # plt.savefig(img_iter, format='png')
        # plt.imsave(str(i).zfill(4) + '.png',C,cmap='gray')
        # os.chdir(root)
        # os.chdir(img_dir)
        # plt.pause(0.001)
        # plt.clf()

    print(time_count/len(img_list))