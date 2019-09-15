import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import time

from glob import glob

from skimage import measure
from scipy import ndimage

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

    img_dir = "img\\output\\background_subtraction_full"
    img_dir2 = "img\\img_binary_search"
    save_dir = "img\\output"

    os.chdir(root)
    os.chdir(img_dir)

    # ---- Generate test image ---- #
    # fig = plt.figure(figsize=(15, np.ceil(15 * A.shape[0] / (A.shape[1] * 2))))

    # A = imageio.imread('test7.bmp')
    # A = (A == 255).astype(int)

    img_list = glob('****.png')
    img_list.sort()

    time_avg = 0
    time_count=0

    for i,img in enumerate(img_list):

        A = imageio.imread(img)  # read the first image in the list
        A = utils.rgb_to_gray(A)  # convert image to 8 bit gray
        A[A > 1] = 1

        start = time.time_ns()

        # ---- Find objects ---- #
        # Filter params
        min_size = 2
        max_size = 400
        max_ratio = 2.5  # height to width ratio

        labels, n_features = ndimage.label(A)  # label image
        label_sizes = ndimage.sum(A, labels, index=range(n_features + 1))

        # # mask all objects under a certain size
        mask = np.bitwise_and(min_size < label_sizes, label_sizes < max_size)
        labels_masked = mask[labels.ravel()].reshape(labels.shape)
        labels[~labels_masked] = 0
        candidates = []

        # filtered_labels = [c.label for c in candidates]
        objects = ndimage.find_objects(labels)
        for j, obj in enumerate(objects):
            if obj is not None:
                label = j + 1
                c = Candidate(label)
                c.size = label_sizes[label]
                center = [int(s) for s in ndimage.center_of_mass(labels, labels, label)]
                c.x = center[1]
                c.y = center[0]
                c.xmin = obj[1].start - 1
                if c.xmin < 0:  c.xmin = 0
                c.xmax = obj[1].stop
                c.ymin = obj[0].start - 1
                if c.ymin < 0:  c.ymin = 0
                c.ymax = obj[0].stop
                # discard if proportions are outside bounds
                width = c.xmax - c.xmin
                height = c.ymax - c.ymin
                if (width / height > max_ratio) | (height / width < 1 / max_ratio):
                    continue
                candidates.append(c)

        time_count += (time.time_ns()-start)/1E9
        time_avg = time_count/(i+1)
        print(time_avg)
        # ---- Plotting ---- #
        plt.imshow(labels_masked, cmap='gray')
        for c in candidates:
            # plt.scatter(c.x,c.y,linewidths=5,color='r',marker='x')
            plt.plot((c.xmin, c.xmin), (c.ymin, c.ymax), color='r', linewidth='1')
            plt.plot((c.xmin, c.xmax), (c.ymax, c.ymax), color='r', linewidth='1')
            plt.plot((c.xmax, c.xmax), (c.ymax, c.ymin), color='r', linewidth='1')
            plt.plot((c.xmax, c.xmin), (c.ymin, c.ymin), color='r', linewidth='1')

        plt.axis('off')
        plt.tight_layout()
        # os.chdir(root)
        # os.chdir(save_dir)
        # plt.savefig(img, bbox_inches = 'tight',pad_inches = 0)
        # os.chdir(root)
        # os.chdir(img_dir)
        plt.clf()
