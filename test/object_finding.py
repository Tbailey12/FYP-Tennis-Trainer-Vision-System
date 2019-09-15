import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import time

from skimage import measure

import utils

class Candidate():
    num_candidates = 0
    candidate_list = []

    def __init__(self, x, y, c_list):
        self.valid = True
        self.x = x
        self.y = y
        self.x_min = x
        self.x_max = x
        self.y_min = y
        self.y_max = y
        self.size = 1
        self.num = Candidate.num_candidates
        self.num_valid = -3
        Candidate.num_candidates += 1
        self.candidate_list = c_list
        self.association = None

    def print_candidates(self):
        # ---- Prints all object candidates ---- #
        for c in self.candidate_list:
            print(c.num)

    def add(self, x, y):
        if x > self.x_max:
            self.x_max = x
        if x < self.x_min:
            self.x_min = x
        if y > self.y_max:
            self.y_max = y
        if y < self.y_min:
            self.y_min = y

        self.size += 1;

    def merge(self, old_candidate):
        if self.association is None:
            old_candidate.association = self.num
        else:
            old_candidate.association = self.association
        if old_candidate.valid:
            old_candidate.valid = False


def check_valid(x, y, xmax, ymax):
    # ---- Ensures checked pixels are not outside bounds of image ---- #
    if (x >= 0) & (y >= 0) & (x < xmax) & (y < ymax):
        return True
    else:
        return False


def check_left(x, y, img):
    if check_valid(x - 1, y, img.shape[1], img.shape[0]):  # pixel is in image
        return img[y, x - 1]
    else:
        return -1


def check_up(x, y, img):
    if check_valid(x, y - 1, img.shape[1], img.shape[0]):  # pixel is in image
        return img[y - 1, x]
    else:
        return -1


def scan(img, candidates):
    A = img - 2
    B = A.copy()
    C = A.copy()

    row_l = 0
    col_l = 0

    for row in range(A.shape[0]):
        for col in range(A.shape[1]):
            A[row_l, col_l] = C[row_l, col_l]
            A[row, col] = 1
            row_l = row
            col_l = col

            if B[row, col] == -1:  # found an object
                left = check_left(col, row, B)
                up = check_up(col, row, B)

                if left < 0:  # no object left
                    if up < 0:  # no object up (new object)
                        c = Candidate(col, row, candidates)  # create a new ball candidate where the object exists
                        candidates.append(c)
                        B[row, col] = c.num
                    else:  # existing object up
                        candidates[up].add(col, row)
                        B[row, col] = candidates[up].num
                else:  # existing object left
                    if up < 0:  # no object up
                        candidates[left].add(col, row)
                        B[row, col] = candidates[left].num
                    else:  # two objects that both exist
                        if candidates[left] == candidates[up]:  # objects are the same
                            candidates[left].add(col, row)
                        else:
                            candidates[up].merge(candidates[left])  # objects are different
                        B[row, col] = candidates[up].num

            # ---- Plotting ---- #
            plt.subplot(1, 2, 1)
            plt.axis('off')
            plt.imshow(A, cmap='gray', vmin=-2)
            plt.subplot(1, 2, 2)
            plt.axis('off')
            plt.imshow(B, cmap='gray', vmin=-1)
            # plt.pause(0.001)
            os.chdir(root)
            os.chdir(save_dir)
            plt.savefig('0' + str(row * A.shape[1] + col).zfill(4) + '.png', format='png')
            os.chdir(root)
            os.chdir(img_dir)
            plt.clf()

    return B


# def find_parent(list, c):
#     if list[c].association:
#         return find_parent(list, list[c].association)
#     else:
#         return list[c].num_valid


def rescan(A, B, candidates):
    A = A - 2
    C = A.copy()

    row_l = 0
    col_l = 0
    valid_candidates = []
    count = 0
    for c in candidates:
        if c.valid:
            c.num_valid = count
            count += 1
            valid_candidates.append(c)

    for c in candidates:
        if not c.valid:
            parent_num = c.association
            c.num_valid = candidates[parent_num].num_valid  # match to parent number in valid_candidates
            if c.x_max > candidates[parent_num].x_max:
                candidates[parent_num].x_max = c.x_max
            if c.x_min < candidates[parent_num].x_min:
                candidates[parent_num].x_min = c.x_min
            if c.y_max > candidates[parent_num].y_max:
                candidates[parent_num].y_max = c.y_max
            if c.y_min < candidates[parent_num].y_min:
                candidates[parent_num].y_min = c.y_min
            candidates[parent_num].size += c.size


    for row in range(A.shape[0]):
        for col in range(A.shape[1]):
            A[row_l, col_l] = C[row_l, col_l]
            A[row, col] = 1
            row_l = row
            col_l = col

            if B[row, col] > -1:  # found an object
                B[row,col] = candidates[B[row, col]].num_valid

            # ---- Plotting ---- #
            plt.subplot(1, 2, 1)
            plt.axis('off')
            plt.imshow(A, cmap='gray', vmin=-2)
            plt.subplot(1, 2, 2)
            plt.axis('off')
            plt.imshow(B, cmap='gray', vmin=-1)
            # plt.pause(0.001)
            os.chdir(root)
            os.chdir(save_dir)
            plt.savefig('1' + str(row * A.shape[1] + col).zfill(4) + '.png', format='png')
            os.chdir(root)
            os.chdir(img_dir)
            plt.clf()
    return valid_candidates

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

    A = imageio.imread('test4.bmp')
    A = (~A == 255).astype(int)
    # plt.imshow(A, cmap='gray')

    # plt.imshow(A-2, cmap='gray')

    start = time.process_time()

    # plt.show()
    # quit()

    fig = plt.figure(figsize=(15, np.ceil(15 * A.shape[0] / (A.shape[1] * 2))))

    # ---- Find objects ---- #
    start = time.process_time()
    potential_candidates = []
    img_seg = scan(A, potential_candidates)  # segment all objects
    candidates = rescan(A, img_seg, potential_candidates)  # join objects based on association

    plt.imshow(img_seg,cmap='gray')
    plt.show()

    print(time.process_time() - start)