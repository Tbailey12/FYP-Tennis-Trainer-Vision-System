import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import time

from skimage import measure
from scipy import ndimage

import utils


class Candidate():
    def __init__(self, label, x, y, size):
        self.valid = True
        self.x = x
        self.y = y
        self.label = label
        self.size = size


if __name__ == "__main__":
    root = utils.get_project_root()

    img_dir = "img\\img_binary_search"
    save_dir = "img\\output"

    os.chdir(root)
    os.chdir(img_dir)

    # ---- Generate test image ---- #
    A = np.random.rand(20, 30)
    A = (A > 0.9).astype(int)
    A = np.array([[0, 0, 0, 1, 0, 1],
                  [0, 1, 1, 1, 1, 0],
                  [0, 1, 0, 0, 1, 0],
                  [1, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1, 1],
                  [1, 1, 0, 0, 0, 0]])

    A = imageio.imread('test8.bmp')
    A = (~A == 255).astype(int)

    A = imageio.imread('test7.bmp')
    A = (A == 255).astype(int)
    # plt.imshow(A, cmap='gray')
    # plt.axis('off')
    # plt.show()

    fig = plt.figure(figsize=(15, np.ceil(15 * A.shape[0] / (A.shape[1] * 2))))

    # ---- Find objects ---- #
    # Filter params
    min_size = 1
    max_size = 400
    max_ratio = 2.5  # height to width ratio

    labels, n_features = ndimage.label(A)  # label image
    label_sizes = ndimage.sum(A, labels, index=range(n_features + 1))


    '''
    need to work out how to:
        mask labels based on size
        find centers of remaining labels
        find bounding box of remaining labels (to determine skew)
    '''
    start = time.time_ns()
    # # mask all objects under a certain size
    mask = np.bitwise_and(min_size < label_sizes, label_sizes < max_size)
    labels_masked = mask[labels.ravel()].reshape(labels.shape)
    labels[~labels_masked] = 0

    candidates = []

    for i, l_size in enumerate(label_sizes):
        if min_size < l_size < max_size:
            centers = [int(s) for s in ndimage.center_of_mass(labels_masked, labels, i)]  # find centre of mass of object
            c = Candidate(i, centers[1], centers[0], l_size)
            candidates.append(c)

    print((time.time_ns() - start) / 1E9)

    objects = ndimage.find_objects(labels)
    filtered_labels = [c.label for c in candidates]
    for label in filtered_labels:
        print(objects[0])
        # print(objects[label][0].start)


    print(n_features)

    # ---- Plotting ---- #
    plt.imshow(labels, cmap='gray')
    for c in candidates:
        plt.scatter(c.x,c.y,linewidths=5,color='r',marker='x')
    plt.axis('off')
    plt.show()
