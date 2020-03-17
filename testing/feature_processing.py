import cv2
import numpy as np
import stereo_calibration_t as s_cal
import os
import time

def rectify_points(max_n, ball_set, camera_matrix, dist_coeffs, R_matrix, P_matrix):
	for frame in range(max_n):
		for ball in range(len(ball_set[frame,1])):
				ball_set[frame,1][ball][X:(Y+1)] = cv2.undistortPoints(ball_set[frame,1][ball][X:(Y+1)], camera_matrix, dist_coeffs, R=R_matrix, P=P_matrix).flatten()

if __name__ == "__main__":
	SIZE = 2
	X = 3
	Y = 4
	WIDTH = 1
	HEIGHT = 2

	# searching 2D array for max
	# test = np.zeros([4,4])
	# test[1,3] = 1
	# test[0,2] = 0.1
	# test[0,3] = 0.4
	# test[2,3] = 1

	# loc = np.where(test==np.amax(test))

	s_calib = s_cal.StereoCal()
	s_calib = s_calib.load_params("stereo_calib.npy")

	left_balls = np.load("left_ball_candidates.npy", allow_pickle=True)
	right_balls = np.load("right_ball_candidates.npy", allow_pickle=True)

	# sort both ball arrays based on frame (first column)
	left_balls = left_balls[left_balls[:,0].argsort()]
	right_balls = right_balls[right_balls[:,0].argsort()]

	# print(left_balls[120,1].astype(int))
	# print(right_balls[120,1].astype(int))

	# testing stereo calibration
	# img = cv2.imread("0000.png",cv2.IMREAD_GRAYSCALE)
	# start = time.time_ns()
	# map1, map2 = cv2.initUndistortRectifyMap(s_calib.cameraMatrix2, s_calib.distCoeffs1, R=s_calib.R2, size=(640,480), newCameraMatrix=s_calib.P2, m1type=cv2.CV_32FC1)
	# print((time.time_ns()-start)/1E6)
	# img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_NEAREST)
	# cv2.imshow("img",img)
	# cv2.waitKey(0)
	# quit()

	# rectify points
	max_n = min(len(left_balls),len(right_balls))
	rectify_points(max_n, left_balls, s_calib.cameraMatrix1, s_calib.distCoeffs1, s_calib.R1, s_calib.P1)
	rectify_points(max_n, right_balls, s_calib.cameraMatrix2, s_calib.distCoeffs2, s_calib.R2, s_calib.P2)

	X_W = 0.2
	Y_W = 0.2
	W_W = 0.5
	H_W = 0.5
	SIZE_W = 1

	# print(left_balls[120,1].astype(int))
	# print(right_balls[120,1].astype(int))

	candidates_3D = []
	matches = []

	# similarity comparison
	sim = np.zeros([100,100],dtype=np.float32)
	for k in range(max_n):
		# reset similarity matching array each frame
		sim = np.zeros([100,100],dtype=np.float32)
		matched = []

		# if there are no ball candidates from either camera, skip frame
		if (len(left_balls[k,1]) == 0) or (len(right_balls[k,1]) == 0):
			candidates_3D.append([k,[]])
			continue

		# loop through left and right candidates
		for i, c_l in enumerate(left_balls[k,1]):
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
					sim[i,j] = 1/calc_b
				else:
					sim[i,j] = np.Inf

			# highest similarity between left candidate and right candidates is used
			r_match = np.where(sim[i]==np.amax(sim[i]))[0][0]
			if r_match in matched:
				break
			matched.append(r_match)

			## -- Triangulate points in 3D space -- ##
			points4d = cv2.triangulatePoints(s_calib.P1,s_calib.P2,\
												(left_balls[k,1][i][X],left_balls[k,1][i][Y]),\
												(right_balls[k,1][r_match][X],right_balls[k,1][r_match][Y]))
				
			## -- Convert homogeneous coordinates to Euclidean space -- ##
			candidates_3D.append([k, np.array([points4d[:3]/points4d[3] for i in points4d[:3]])])

	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('x (m)')
	ax.set_ylabel('y (m)')
	ax.set_zlabel('z (m)')
	ax.set_xlim(-2, 2)
	ax.set_ylim(0, 25)
	ax.set_zlim(0, 1)

	os.chdir("plots")

	for frame, point in candidates_3D:
		if len(point) > 0 and point.item(2)>0:
			ax.scatter(xs=point.item(0),ys=point.item(2),zs=-point.item(1)+0.5)
		plt.savefig(f"{frame:04d}.png")
	plt.show()