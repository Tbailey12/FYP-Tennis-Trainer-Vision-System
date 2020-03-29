import cv2
import numpy as np
import stereo_calibration_t as s_cal
import os
import time
import timeit

def rectify_points(frame_set, camera_matrix, dist_coeffs, R_matrix, P_matrix):
	for frame in frame_set:
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

	s_calib = s_cal.StereoCal()
	s_calib.load_params("0.8524stereo_calib.npy")

	left_balls = np.load("left_ball_candidates.npy", allow_pickle=True)
	right_balls = np.load("right_ball_candidates.npy", allow_pickle=True)

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
	left_candidates = 1000*[None]
	right_candidates = 1000*[None]
	# for f, frame in enumerate(left_balls):
	# 	left_candidates[f] = frame[1]
	# for f, frame in enumerate(right_balls):
	# 	right_candidates[f] = frame[1]
	left_candidates = [x for x in left_candidates if x is not None]
	right_candidates = [x for x in right_candidates if x is not None]

	# -- Make both lists the same length -- #
	min_length = min(len(left_candidates), len(right_candidates))
	left_candidates = left_candidates[:min_length]
	right_candidates = right_candidates[:min_length]


	candidates_3D = 1000*[[]]

	rectify_points(left_candidates, s_calib.cameraMatrix1, s_calib.distCoeffs1, s_calib.R1, s_calib.P1)
	rectify_points(right_candidates, s_calib.cameraMatrix2, s_calib.distCoeffs2, s_calib.R2, s_calib.P2)
	
	# print(left_candidates[139])
	# print(right_candidates[139])

	pointsL = np.array([[[460.14,255.85]],[[518.33,254.16]],[[577.87,251.63]],
						[[461.83,314.88]],[[520.86,312.35]],[[579.47,310.24]],
						[[464.36,373.91]],[[523.39,371.80]],[[581.58,369.69]]], dtype=np.float32)

	pointsR = np.array([[[35.30,257.67]],[[93.75,257.67]],[[152.56,257.29]],
						[[35.30,317.25]],[[93.75,316.87]],[[152.57,316.11]],
						[[35.30,375.69]],[[94.12,376.07]],[[152.57,376.07]]], dtype=np.float32)

	# [0.0794484, 0.014571001, 0.49833423] [0.102630295, 0.014499336, 0.49964216]
	# [0.07952101, 0.037906643, 0.4984369]
	# [0.07982413, 0.06094061, 0.49816146]


	pointsLu = cv2.undistortPoints(pointsL, s_calib.cameraMatrix1, s_calib.distCoeffs1, R=s_calib.R1, P=s_calib.P1)
	pointsRu = cv2.undistortPoints(pointsR, s_calib.cameraMatrix2, s_calib.distCoeffs2, R=s_calib.R2, P=s_calib.P2)
	
	points4d = cv2.triangulatePoints(s_calib.P1, s_calib.P2, pointsLu[1][0], pointsRu[1][0]).flatten()
	points3d = [i/points4d[3] for i in points4d[:3]]
	

	for f, frame in enumerate(left_candidates):
		if len(left_candidates[f]) == 0 or len(right_candidates[f]) == 0:
			candidates_3D[f] = []
			continue
		else:
			candidates = []
			for i_l, c_l in enumerate(left_candidates[f]):
				max_sim = 0
				for j_r, c_r in enumerate(right_candidates[f]):
					width_r = c_l[WIDTH]/c_r[WIDTH]
					size_r = c_l[SIZE]/c_r[SIZE]
					height_r = c_l[HEIGHT]/c_r[HEIGHT]

					if width_r<1: width_r=1/width_r
					if size_r<1: size_r=1/size_r
					if height_r<1: height_r=1/height_r


					calc_b = (	SIZE_W*(size_r)**2 +\
									W_W*(width_r)**2 +\
									H_W*(height_r)**2 +\
									X_W*(c_l[X]-c_r[X])**2 +\
									Y_W*(c_l[Y]-c_r[Y])**2 )

					if calc_b != 0:
						sim = 1/calc_b
					else:
						sim = np.Inf
					
					if sim>max_sim:
						r_match = j_r
						max_sim = sim

				points4d = cv2.triangulatePoints(s_calib.P1, s_calib.P2, c_l[X:Y+1], right_candidates[f][r_match][X:Y+1]).flatten()
				points3d = [i/points4d[3] for i in points4d[:3]]
				candidates.append(points3d)

			candidates_3D[f] = candidates

	## -- Plot points -- ##
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('x (m)')
	ax.set_ylabel('y (m)')
	ax.set_zlabel('z (m)')
	ax.set_xlim(-1, 1)
	ax.set_ylim(0, 15)
	ax.set_zlim(-1, 2)

	x = 0
	y = 1
	z = 2

	for f, candidates in enumerate(candidates_3D):
		for candidate in candidates:
			ax.scatter(xs=candidate[0]-(25E-2/2),ys=candidate[2],zs=-candidate[1])
	plt.show()
	quit()

	# similarity comparison
	# sim = np.zeros([100],dtype=np.float32)
	for k in range(max_n):
		# reset similarity matching array each frame
		# sim = np.zeros([100],dtype=np.float32)
		# matched = []

		# if there are no ball candidates from either camera, skip frame
		if (len(left_balls[k,1]) == 0) or (len(right_balls[k,1]) == 0):
			candidates_3D.append([k,[]])
			continue

		start = timeit.default_timer()
		# loop through left and right candidates
		for i, c_l in enumerate(left_balls[k,1]):
			max_sim = 0
			for j, c_r in enumerate(right_balls[k,1]):
				width_r = c_l[WIDTH]/c_r[WIDTH]
				size_r = c_l[SIZE]/c_r[SIZE]
				height_r = c_l[HEIGHT]/c_r[HEIGHT]

				if width_r<1: width_r=1/width_r
				if size_r<1: size_r=1/size_r
				if height_r<1: height_r=1/height_r


				calc_b = (	SIZE_W*(size_r)**2 +\
								W_W*(width_r)**2 +\
								H_W*(height_r)**2 +\
								X_W*(c_l[X]-c_r[X])**2 +\
								Y_W*(c_l[Y]-c_r[Y])**2 )
				if calc_b != 0:
					sim = 1/calc_b
				else:
					sim = np.Inf
				
				if sim>max_sim:
					r_match = j
					max_sim = sim

			# highest similarity between left candidate and right candidates is used
			# r_match = np.where(sim[i]==np.amax(sim[i]))[0][0]
			# if r_match in matched:
			# 	break
			# matched.append(r_match)

			## -- Triangulate points in 3D space -- ##
			points4d = cv2.triangulatePoints(s_calib.P1,s_calib.P2,\
												(left_balls[k,1][i][X],left_balls[k,1][i][Y]),\
												(right_balls[k,1][r_match][X],right_balls[k,1][r_match][Y]))
				
			## -- Convert homogeneous coordinates to Euclidean space -- ##
			candidates_3D.append([k, np.array([points4d[:3]/points4d[3] for i in points4d[:3]])])


	print(candidates_3D[176])
	quit()
	print((timeit.default_timer()-start))

	np.save('candidates_3D.npy', candidates_3D)

	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('x (m)')
	ax.set_ylabel('y (m)')
	ax.set_zlabel('z (m)')
	ax.set_xlim(-2, 2)
	ax.set_ylim(0, 25)
	ax.set_zlim(-2, 1)

	# os.chdir("plots")

	for frame, point in candidates_3D:
		if len(point) > 0 and point.item(2)>0:
			ax.scatter(xs=point.item(0),ys=point.item(2),zs=point.item(1))
		# plt.savefig(f"{frame:04d}.png")
	plt.show()