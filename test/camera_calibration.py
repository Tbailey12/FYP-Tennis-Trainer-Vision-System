import cv2 as cv
from glob import glob
import numpy as np

import utils
import os

import matplotlib.pyplot as plt


def process_image(f):
    img = cv.imread(f, 0)  # read image
    if img is None:
        print("failed to load: ", f)
        return None

    # check to ensure that the image matches the width/height of the initial image
    assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ..." % (img.shape[1], img.shape[0]))

    found, corners = cv.findChessboardCorners(img, pattern_size)
    if found:
        # term defines when to stop refinement of subpixel coords
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
        cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        # draw corners on the image depending on draw_corners flag
        if draw_corners_flg:
            img_c = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            cv.drawChessboardCorners(img_c, pattern_size, corners, found)
            plt.imshow(img_c)
            plt.show()

    if not found:
        print('chessboard not found')
        return None

    # print message if file contains chessboard
    print('%s... OK' % f)
    return (corners.reshape(-1, 2), pattern_points)


if __name__ == "__main__":

    square_size = 1
    pattern_size = (9, 6)  # number of points (where the black and white intersects)
    draw_corners_flg = False

    root = utils.get_project_root()
    img_dir = "img\\calib"

    os.chdir(root)
    os.chdir(img_dir)

    # -- Read in all calibration images -- #
    img_list = glob('left**.jpg')
    img_list.sort()

    # -- Setup point lists -- #
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)  # x,y,z for all points in image
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)  # p.u for all point positions
    pattern_points *= square_size  # scale by square size for point coords

    obj_points = []
    img_points = []
    h, w = cv.imread(img_list[0], cv.IMREAD_GRAYSCALE).shape[:2]  # get height and width of image

    # find the chessboard points in all images
    chessboards = [process_image(f) for f in img_list]
    chessboards = [x for x in chessboards if x is not None]

    # split the chessboards into image points and object points
    for(corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)

    # -- Calculate camera distortion parameters -- #
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w,h), None, None)

    # -- Undistort an image -- #
    img_g = cv.imread(img_list[0], cv.IMREAD_GRAYSCALE)
    img_ud = cv.undistort(img_g, camera_matrix, dist_coefs, None, camera_matrix)

    plt.subplot(121)
    plt.title('Original')
    plt.imshow(img_g, cmap='gray')
    plt.axis('off')
    plt.subplot(122)
    plt.title('Undistorted')
    plt.imshow(img_ud, cmap='gray')
    plt.axis('off')
    plt.show()
