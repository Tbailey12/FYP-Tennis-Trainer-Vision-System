import cv2 as cv
from glob import glob
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import consts as c
import camera_calibration as cal

class StereoCal(object):
	def __init__(self, rms, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F, R1, R2, P1, P2, Q, validPixROI1, validPixROI2):
		self.rms = rms
		self.cameraMatrix1 = cameraMatrix1
		self.distCoeffs1 = distCoeffs1
		self.cameraMatrix2 = cameraMatrix2
		self.distCoeffs2 = distCoeffs2
		self.R = R
		self.T = T
		self.E = E
		self.F = F
		self.R1 = R1
		self.R2 = R2
		self.P1 = P1
		self.P2 = P2
		self.Q = Q
		self.validPixROI1 = validPixROI1
		self.validPixROI2 = validPixROI2

	def save_params(filename):
	    with open(filename, 'wb') as f:
	        np.savez(f, rms = self.rms, cameraMatrix1 = self.cameraMatrix1, distCoeffs1 = self.distCoeffs1, cameraMatrix2 = self.cameraMatrix2, distCoeffs2 = self.distCoeffs2, R = self.R, T = self.T, E = self.E, F = self.F, R1 = self.R1, R2 = self.R2, P1 = self.P1, P2 = self.P2, Q = self.Q, validPixROI1 = self.validPixROI1, validPixROI2 = self.validPixROI2)

	def load_params(filename):
	    with open(filename, 'rb') as f:
	        myfile = np.load(f)
	        cam_cal = CamCal(myfile['rms'], myfile['cameraMatrix1'], myfile['distCoeffs1'], myfile['cameraMatrix2'], myfile['distCoeffs2'], myfile['R'], myfile['T'], myfile['E'], myfile['F'], myfile['R1'], myfile['R2'], myfile['P1'], myfile['P2'], myfile['Q'], myfile['validPixROI1'], myfile['validPixROI2'])
	        return cam_cal
	    return None

def process_image(img_data, pattern_points):
    n_frame, img = img_data

    # check to ensure that the image matches the width/height of the initial image
    # assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ..." % (img.shape[1], img.shape[0]))

    found, corners = cv.findChessboardCorners(img, c.pattern_size)
    if found:
        # term defines when to stop refinement of subpixel coords
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
        cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

    if not found:
        print('chessboard not found')
        return None

    # print message if file contains chessboard
    # print('%s... OK' % n_frame)
    return (n_frame, corners.reshape(-1, 2), pattern_points)

def generate_pattern_points():
    # -- Setup point lists -- #
    pattern_points = np.zeros((np.prod(c.pattern_size), 3), np.float32)  # x,y,z for all points in image
    pattern_points[:, :2] = np.indices(c.pattern_size).T.reshape(-1, 2)  # p.u for all point positions
    pattern_points *= c.square_size  # scale by square size for point coords
    return pattern_points


def find_chessboards(img_data):
    # frames = []
    # obj_points = []
    # img_points = []

    pattern_points = generate_pattern_points()

    # find the chessboard points in all images
    chessboards = [process_image(img, pattern_points) for img in img_data]
    chessboards = [x for x in chessboards if x is not None]

    # split the chessboards into image points and object points
    # for(n_frame, corners, pattern_points) in chessboards:
    #     frames.append(n_frame)
    #     img_points.append(corners)
    #     obj_points.append(pattern_points)

    return chessboards

## -- Checks whether chessboards for a given frame are in both arrays -- ##
## - removes the board from the array if there is no matching chessboard
def validate_chessboards(left_chessboards, right_chessboards):
	if len(left_chessboards) > 0 and len(right_chessboards) > 0:
		for i, chessboard_L in enumerate(left_chessboards):
			if chessboard_L[0] == right_chessboards[i][0]:
				# chessboard was found in both cameras
				continue
			elif chessboard_L[0] > right_chessboards[i][0]:
				# the chessboard was found in the right cam but not the left
				print('missing left chessboard')
				del right_chessboards[i]
			elif chessboard_L[0] < right_chessboards[i][0]:
				# the chessboard was found in the left cam but not the right
				print('missing right chessboard')
				del left_chessboards[i]

def calibrate_stereo(left_chessboards, right_chessboards, left_cam, right_cam, size):
	## -- Separate chessboards into arrays -- ##
	object_points = []
	left_image_points = []
	right_image_points = []

	for (n_frame, image_points, obj_points) in left_chessboards:
		left_image_points.append(image_points)
		object_points.append(obj_points)

	for (n_frame, image_points, obj_points) in right_chessboards:
		right_image_points.append(image_points)

	## -- Perform stereo calibration -- ##
	term_crit = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-6)
	if len(left_chessboards)>8:
		RMS, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv.stereoCalibrate(
								object_points, left_image_points, right_image_points, left_cam.camera_matrix, 
								left_cam.dist_coefs, right_cam.camera_matrix, right_cam.dist_coefs, 
								size, criteria=term_crit ,flags=cv.CALIB_FIX_INTRINSIC)
	
	else:
		print('there is not enough chessboard views for calibration, please repeat')

	return RMS, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F


def main():
	img_folder = 'calib_S'
	root = os.getcwd()
	img_list = os.listdir(root+'/'+img_folder)

	left_img_data = []
	right_img_data = []

	size = 0

	## -- Load all images in for testing -- ##

	for img_name in img_list:
		if c.LEFT_CLIENT in img_name:
			img = cv.imread(root+'/'+img_folder+'/'+img_name, cv.IMREAD_GRAYSCALE)
			frame_n = img_name[len(c.LEFT_CLIENT):-4]
			left_img_data.append((frame_n, img))
			(h,w) = img.shape[:2]
			size = (w,h)

		elif c.RIGHT_CLIENT in img_name:
			img = cv.imread(root+'/'+img_folder+'/'+img_name, cv.IMREAD_GRAYSCALE)
			frame_n = img_name[len(c.RIGHT_CLIENT):-4]
			right_img_data.append((frame_n, img))

	## -- Find chessboards in images -- ##
	left_chessboards = find_chessboards(left_img_data)
	right_chessboards = find_chessboards(right_img_data)

	validate_chessboards(left_chessboards, right_chessboards)

	## -- Load camera data -- ##
	left_cam = cal.load_params("calib_L.npy")
	right_cam = cal.load_params("calib_R.npy")

	RMS, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = calibrate_stereo(
								left_chessboards, right_chessboards, left_cam, right_cam, size)
	
	## -- Obtain stereo rectification projection matrices -- ##
	R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(cameraMatrix1, distCoeffs1,
								cameraMatrix2, distCoeffs2, size, R, T)

	# Lx,Ly = (253.0, 218.0)
	# Rx,Ry = (49.5, 202.8)

	# bottle of sunscreen [0,0.05,3]
	Lx,Ly = (378.5, 304.5)
	Rx,Ry = (259.6, 298.6)

	# Lx,Ly = (320+20, 240)
	# Rx,Ry = (320-20, 240)

	LPointsd = np.array([[Lx,Ly]], dtype=np.float32).reshape(-1,1,2)
	RPointsd = np.array([[Rx,Ry]], dtype=np.float32).reshape(-1,1,2)


	## -- Undistort points based on camera matrix and rectification projection -- ##
	LPointsu = cv.undistortPoints(LPointsd, cameraMatrix1, distCoeffs1, R=R1, P=P1)
	RPointsu = cv.undistortPoints(RPointsd, cameraMatrix2, distCoeffs2, R=R2, P=P2)

	## -- Triangulate points in 3D space -- ##
	points4d = cv.triangulatePoints(P1,P2,LPointsu,RPointsu)

	## -- Convert homogeneous coordinates to Euclidean space -- ##
	points3d = np.array([i/points4d[3] for i in points4d[:3]])
	print(points3d)


if __name__ == "__main__":
	main()