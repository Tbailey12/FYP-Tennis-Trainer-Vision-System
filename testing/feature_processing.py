import cv2
import numpy as np
import stereo_calibration_t as s_cal
import os
import time
import timeit

ROOT_P = 'D:\\documents\\local uni\\FYP\\code'

def rectify_points(frame_set, camera_matrix, dist_coeffs, R_matrix, P_matrix):
	for frame in frame_set:
		if frame is not None:
			for candidate in frame:
				if candidate is not []:
					candidate[X:Y+1] = cv2.undistortPoints(candidate[X:Y+1], camera_matrix, dist_coeffs, R=R_matrix, P=P_matrix)

if __name__ == "__main__":
	SIZE = 2
	X = 3
	Y = 4
	WIDTH = 0
	HEIGHT = 1

	RESOLUTION = (640,480)
	w,h = RESOLUTION

	# searching 2D array for max
	# test = np.zeros([4,4])
	# test[1,3] = 1
	# test[0,2] = 0.1
	# test[0,3] = 0.4
	# test[2,3] = 1

	# loc = np.where(test==np.amax(test))

	os.chdir(ROOT_P + '\\' + 'img\\inside_tests')

	s_calib = s_cal.StereoCal()
	# s_calib.load_params("0.4485stereo_calib.npy")
	# s_calib.load_params("0.0410stereo_calib.npy")
	s_calib.load_params("0.3071stereo_calib.npy")


	left_balls = np.load("left_ball_candidates.npy", allow_pickle=True)
	right_balls = np.load("right_ball_candidates.npy", allow_pickle=True)

	print(left_balls[16,1])
	print(left_balls[17,1])
	print(right_balls[16,1])
	print(right_balls[17,1])


	# sort both ball arrays based on frame (first column)
	# left_balls = left_balls[left_balls[:,0].argsort()]
	# right_balls = right_balls[right_balls[:,0].argsort()]

	
	# testing stereo calibration
	# img = cv2.imread("0000.png",cv2.IMREAD_GRAYSCALE)
	# start = time.time_ns()
	# map1, map2 = cv2.initUndistortRectifyMap(s_calib.cameraMatrix2, s_calib.distCoeffs1, R=s_calib.R2, size=(640,480), newCameraMatrix=s_calib.P2, m1type=cv2.CV_32FC1)
	# print((time.time_ns()-start)/1E6)
	# img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_NEAREST)
	# cv2.imshow("img",img)
	# cv2.waitKey(0)
	# quit()

	# print(left_balls[160,1])
	# # rectify points
	# max_n = min(len(left_balls),len(right_balls))
	# rectify_points(max_n, left_balls, s_calib.cameraMatrix1, s_calib.distCoeffs1, s_calib.R1, s_calib.P1)
	# rectify_points(max_n, right_balls, s_calib.cameraMatrix2, s_calib.distCoeffs2, s_calib.R2, s_calib.P2)

	# print(left_balls[160,1][0][X:Y+1])
	# print(right_balls[160,1][0][X:Y+1])

	# A = cv2.triangulatePoints(s_calib.P1,s_calib.P2, left_balls[160,1][0][X:Y+1], right_balls[160,1][0][X:Y+1])
	# B = np.array([i/A[3] for i in A[:3]])
	# print(B)

	# X_W = 0.2
	# Y_W = 1
	# W_W = 0.5
	# H_W = 0.5
	X_W = 0
	Y_W = 1
	W_W = 0.5
	H_W = 0.5
	SIZE_W = 0.75

	# -- Modify both candidate lists so that frames are list indices -- #
	left_candidates = len(left_balls)*[None]
	right_candidates = len(right_balls)*[None]
	for f, frame in enumerate(left_balls):
		left_candidates[f] = frame[1]
	for f, frame in enumerate(right_balls):
		right_candidates[f] = frame[1]
	# left_candidates = [x for x in left_candidates if x is not None]
	# right_candidates = [x for x in right_candidates if x is not None]

	# -- Make both lists the same length -- #
	min_length = min(len(left_balls), len(right_balls))
	left_candidates = left_candidates[:min_length]
	right_candidates = right_candidates[:min_length]

	candidates_3D = min_length*[[]]

	rectify_points(left_candidates, s_calib.cameraMatrix1, s_calib.distCoeffs1, s_calib.R1, s_calib.P1)
	rectify_points(right_candidates, s_calib.cameraMatrix2, s_calib.distCoeffs2, s_calib.R2, s_calib.P2)
	
	# print(left_candidates[139])
	# print(right_candidates[139])

	# pointsL = np.array([[[460.14,255.85]],[[518.33,254.16]],[[577.87,251.63]],
	# 					[[461.83,314.88]],[[520.86,312.35]],[[579.47,310.24]],
	# 					[[464.36,373.91]],[[523.39,371.80]],[[581.58,369.69]]], dtype=np.float32)

	# pointsR = np.array([[[35.30,257.67]],[[93.75,257.67]],[[152.56,257.29]],
	# 					[[35.30,317.25]],[[93.75,316.87]],[[152.57,316.11]],
	# 					[[35.30,375.69]],[[94.12,376.07]],[[152.57,376.07]]], dtype=np.float32)

	# [0.0794484, 0.014571001, 0.49833423] [0.102630295, 0.014499336, 0.49964216]
	# [0.07952101, 0.037906643, 0.4984369]
	# [0.07982413, 0.06094061, 0.49816146]


	# pointsLu = cv2.undistortPoints(pointsL, s_calib.cameraMatrix1, s_calib.distCoeffs1, R=s_calib.R1, P=s_calib.P1)
	# pointsRu = cv2.undistortPoints(pointsR, s_calib.cameraMatrix2, s_calib.distCoeffs2, R=s_calib.R2, P=s_calib.P2)
	
	# points4d = cv2.triangulatePoints(s_calib.P1, s_calib.P2, pointsLu[1][0], pointsRu[1][0]).flatten()
	# points3d = [i/points4d[3] for i in points4d[:3]]


	theta = -10
	theta = theta*np.pi/180

	Rx = np.array(	[[1,		0,				0, 				0],
					[0,			np.cos(theta),	-np.sin(theta),	0],
					[0,			np.sin(theta),	np.cos(theta),	0],
					[0,			0,				0,				1]], dtype=np.float32)

	Tz = np.array([	[1, 0, 0, -(25E-2/2)],
					[0,	1, 0, 0],
					[0, 0, 1, 35E-2],
					[0, 0, 0, 1]], dtype=np.float32)

	Rx = Rx.dot(Tz)

	DISP_Y = 30

	for f, frame in enumerate(left_candidates):
		print(f)
		if len(left_candidates[f]) == 0 or len(right_candidates[f]) == 0:
			candidates_3D[f] = []
			continue
		else:
			candidates = []
			for i_l, c_l in enumerate(left_candidates[f]):
				max_sim = 0
				for j_r, c_r in enumerate(right_candidates[f]):
					if abs(c_l[Y]-c_r[Y]) < DISP_Y:

						width_r = c_l[WIDTH]/c_r[WIDTH]
						size_r = c_l[SIZE]/c_r[SIZE]
						height_r = c_l[HEIGHT]/c_r[HEIGHT]

						if width_r<1: width_r=1/width_r
						if size_r<1: size_r=1/size_r
						if height_r<1: height_r=1/height_r

						calc_b = ((size_r)**2 + (width_r)**2 + (height_r)**2)

						if calc_b != 0:
							sim = 3/calc_b
						else:
							sim = np.Inf
				
						if sim>max_sim:
							r_match = j_r
							max_sim = sim
				if max_sim > 0.1:
					print(max_sim)
					# print(c_l[X:Y+1])
					# print(right_candidates[f][r_match][X:Y+1])
					points4d = cv2.triangulatePoints(s_calib.P1, s_calib.P2, left_candidates[f][i_l][X:Y+1], right_candidates[f][r_match][X:Y+1]).flatten()
					points3d = [i/points4d[3] for i in points4d[:3]]
					points3d_shift = [points3d[0], points3d[2], -points3d[1], 1]
					points3d_shift = Rx.dot(points3d_shift)
					candidates.append(points3d_shift)

			candidates_3D[f] = candidates
			print(candidates)
			# if f>16:
			# 	break

	np.save('candidates_3D.npy', candidates_3D)

	## -- Plot points -- ##
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('x (m)')
	ax.set_ylabel('y (m)')
	ax.set_zlabel('z (m)')
	ax.set_xlim(-11E-1/2, 11E-1/2)
	ax.set_ylim(0, 24E-1)
	ax.set_zlim(0, 2E-1)

	os.chdir("plots")

	count = 0
	for f, candidates in enumerate(candidates_3D):
		for candidate in candidates:
			ax.scatter(xs=candidate[0],ys=candidate[1],zs=candidate[2])
			count+=1
		# plt.savefig(f"{f:04d}.png")


	print(count)
	plt.show()
	quit()