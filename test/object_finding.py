import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import time

import utils

class candidate():
    num_candidates = 0
    candidate_list = []

    def __init__(self, x, y, img, list):
        self.valid = True
        self.img = img
        self.root_x = x
        self.root_y = y
        self.x = x
        self.y = y
        self.x_min = x
        self.x_max = x
        self.y_min = y
        self.y_max = y
        self.size = 1
        self.num = candidate.num_candidates
        candidate.num_candidates += 1
        self.candidate_list = list

    def explore(self):
        self.check_self()

    def check_self(self):
        # ---- Checks the current pixel ---- #
        if self.img[self.y, self.x] == -1:
            self.draw_loc()  # for debugging
            self.check_right()



    def check_right(self):
        # ---- Checks to the right of the current pixel ---- #
        x_l = self.x
        y_l = self.y
        if self.check_valid(self.x + 1, self.y):  # if pixel is in image
            if self.img[self.y, self.x + 1] == -1:  # if pixel is a new object
                self.x_max = self.x + 1  # new rightmost pixel
                self.x = self.x + 1  # move right
                self.draw_loc()  # for debugging
                self.size += 1  # increase size
                self.check_right()
            elif (self.img[self.y, self.x + 1] > 0) & (
                    self.img[self.y, self.x + 1] != self.num):  # if pixel is in another object
                self.candidate_list[self.img[self.y, self.x + 1]].valid = False  # set previous object to false
                self.x_max = self.x + 1  # new rightmost pixel
                self.x = self.x + 1  # move right
                self.draw_loc()  # for debugging
                self.size += 1  # increase size
                self.check_right()
        self.set_x_y(x_l, y_l)
        self.check_up()
        self.check_down()
        self.check_left()

    def check_down(self):
        # ---- Checks below the current pixel ---- #
        x_l = self.x
        y_l = self.y
        if self.check_valid(self.x, self.y + 1):  # if pixel is in image
            if self.img[self.y + 1, self.x] == -1:  # if pixel is a new object
                self.y_max = self.y + 1
                self.y = self.y + 1
                self.draw_loc()  # for debugging
                self.size += 1
                self.check_right()
            elif (self.img[self.y + 1, self.x] > 0) & (
                    self.img[self.y + 1, self.x] != self.num):  # if pixel is in another object
                self.candidate_list[self.img[self.y + 1, self.x]].valid = False  # set previous object to false
                self.y_max = self.y + 1  # new bottom pixel
                self.y = self.y + 1
                self.draw_loc()  # for debugging
                self.size += 1
                self.check_right()
        # self.set_x_y(x_l, y_l)

    def check_left(self):
        x_l = self.x
        y_l = self.y
        if self.check_valid(self.x - 1, self.y):  # if pixel is in image
            if self.img[self.y, self.x - 1] == -1:  # if pixel is a new object
                self.x_min = self.x - 1
                self.x = self.x - 1
                self.draw_loc()  # for debugging
                self.size += 1
                self.check_right()
            elif self.img[self.y, self.x - 1] > 0:  # if pixel belongs to another object
                self.candidate_list[self.img[self.y, self.x - 1]].valid = False  # set previous object to false
                self.x_min = self.x - 1
                self.x = self.x - 1
                self.draw_loc()  # for debugging
                self.size += 1
                self.check_right()
        # self.set_x_y(x_l, y_l)

    def check_up(self):
        x_l = self.x
        y_l = self.y
        # ---- Used for propogating back up the branches of another object ---- #
        if self.check_valid(self.x, self.y - 1):  # if pixel is in image
            if self.img[self.y - 1, self.x] == -1:  # if pixel is a new object
                self.y_min = self.y - 1
                self.y = self.y - 1
                self.draw_loc()  # for debugging
                self.size += 1
                self.check_right()
            elif self.img[self.y - 1, self.x] > 0:  # if pixel belongs to another object
                self.candidate_list[self.img[self.y - 1, self.x]].valid = False  # set previous object to false
                self.y_min = self.y - 1
                self.y = self.y - 1
                self.draw_loc()  # for debugging
                self.size += 1
                self.check_right()
        # self.set_x_y(x_l, y_l)

    def return_to_root(self):
        self.x = self.root_x
        self.y = self.root_y

    def set_x_y(self, x, y):
        self.x = x
        self.y = y

    def print_candidates(self):
        # ---- Prints all object candidates ---- #
        for c in self.candidate_list:
            print(c.num)

    def draw_loc(self):
        self.img[self.y, self.x] = self.num

    def check_valid(self, x, y):
        # ---- Ensures checked pixels are not outside bounds of image ---- #
        if (x >= 0) & (y >= 0) & (x < self.img.shape[1]) & (y < self.img.shape[0]):
            return True
        else:
            return False


if __name__ == "__main__":
    root = utils.get_project_root()

    img_dir = "img\\img_binary_search"
    save_dir = "img\\output"

    os.chdir(root)
    os.chdir(img_dir)

    # ---- Generate test image ---- #
    A = np.random.rand(20, 30)
    A = (A > 0.9).astype(int)
    # A = np.array([[0, 0, 0, 1, 0, 1],
    #               [0, 1, 1, 1, 1, 0],
    #               [0, 1, 0, 0, 1, 0],
    #               [1, 0, 0, 1, 0, 0],
    #               [0, 0, 0, 1, 1, 1],
    #               [1, 1, 0, 0, 0, 0]])

    A = imageio.imread('test3.bmp')
    A = (~A == 255).astype(int)

    A = A - 2
    B = A.copy()
    C = A.copy()

    plt.figure(figsize=(15, np.ceil(15 * A.shape[0] / (A.shape[1] * 2))))

    # ---- Find objects ---- #
    candidates = []
    row_l = 0
    col_l = 0
    start = time.process_time()
    for row in range(A.shape[0]):
        for col in range(A.shape[1]):
            A[row_l, col_l] = C[row_l, col_l]
            A[row, col] = 1
            row_l = row
            col_l = col
            if B[row, col] == -1:
                c = candidate(col, row, B, candidates)  # create a new ball candidate where the object exists
                candidates.append(c)
                c.explore()

            # ---- Plotting ---- #
            plt.subplot(1, 2, 1)
            plt.axis('off')
            plt.imshow(A, cmap='gray', vmin=-2)
            plt.subplot(1, 2, 2)
            plt.axis('off')
            plt.imshow(B, cmap='gray', vmin=-1)

            # os.chdir(root)
            # os.chdir(save_dir)
            # plt.savefig(str(row*A.shape[1]+col).zfill(4)+'.png', format='png')
            # os.chdir(root)
            # os.chdir(img_dir)

            plt.pause(0.001)
            plt.clf()
    print(time.process_time() - start)
    # plt.subplot(2, 2, 2)
    # plt.imshow(C, cmap='gray', vmin=0, vmax=1)
    # plt.subplot(2, 2, 3)
    # plt.imshow(A, cmap='gray', vmin=0, vmax=1)
    # plt.subplot(2, 2, 4)
    # plt.imshow(B, cmap='gray', vmin=0)
    # plt.axis('off')
    # plt.show()
