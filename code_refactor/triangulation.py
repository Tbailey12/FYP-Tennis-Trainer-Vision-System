import numpy as np
import cv2

import consts as c

def get_rotation_matrix():
	Rx = np.array(  [[1,		0,					  0,					  0],
					[0,		 np.cos(c.CAM_ANGLE),	-np.sin(c.CAM_ANGLE),   0],
					[0,		 np.sin(c.CAM_ANGLE),	np.cos(c.CAM_ANGLE),	0],
					[0,		 0,					  0,					  1]], dtype=np.float32)
	return Rx

def get_translation_matrix():
	Tz = np.array([ [1, 0, 0, -(c.CAM_BASELINE/2)],
				[0, 1, 0, 0],
				[0, 0, 1, c.CAM_HEIGHT],
				[0, 0, 0, 1]], dtype=np.float32)
	return Tz

def get_projection_matrix():
	Rx = get_rotation_matrix()
	Tz = get_translation_matrix()

	return Rx.dot(Tz)

def triangulate_points(ball_candidate_dict, stereo_calib):
	Px = get_projection_matrix()

	num_frames = max(list(ball_candidate_dict[c.LEFT_CLIENT].keys()))+1
	points_3d = num_frames*[[]]

	for f in ball_candidate_dict[c.LEFT_CLIENT]:
		try:
			frame_points_3d = []
			left_candidates = ball_candidate_dict[c.LEFT_CLIENT][f]
			right_candidates = ball_candidate_dict[c.RIGHT_CLIENT][f]

			for left_cand in left_candidates:
				right_cand = find_most_similar(left_cand, right_candidates)
				
				if right_cand is None:
					continue
				else:
					frame_points_3d.append(calc_3d_point(stereo_calib, Px, left_cand, right_cand))

			points_3d[f] = frame_points_3d

		except KeyError:
			continue

	return points_3d

def calc_3d_point(stereo_calib, Px, left_cand, right_cand):
	point_4d = cv2.triangulatePoints(stereo_calib.P1, stereo_calib.P2, \
				left_cand[c.X_COORD:c.Y_COORD+1], right_cand[c.X_COORD:c.Y_COORD+1]).flatten()
	point_3d = [i/point_4d[3] for i in point_4d[:3]]
	point_3d_shift = [point_3d[0], point_3d[2], -point_3d[1], 1]
	point_3d_shift = Px.dot(point_3d_shift)[:3]

	return point_3d_shift

def find_most_similar(left_cand, right_candidates):
	max_sim = 0
	for right_cand in right_candidates:
		sim = calc_similarity(left_cand, right_cand)

		if sim is None:
			continue
		elif sim > max_sim:
			r_match = right_cand
			max_sim = sim

	if max_sim > c.SIM_THRESH:
		return r_match
	else:
		return None

def calc_similarity(left_cand, right_cand):
	if abs(left_cand[c.Y_COORD]-right_cand[c.Y_COORD]) < c.DISP_Y:

		# calculate ratio between left and right candidate
		width_r = left_cand[c.WIDTH]/right_cand[c.WIDTH]
		size_r = left_cand[c.SIZE]/right_cand[c.SIZE]
		height_r = left_cand[c.HEIGHT]/right_cand[c.HEIGHT]

		# ensure all ratios are > 1
		if width_r<1: width_r=1/width_r
		if size_r<1: size_r=1/size_r
		if height_r<1: height_r=1/height_r

		if width_r>c.SIM_WR_THRESH or size_r>c.SIM_SR_THRESH or height_r>c.SIM_HR_THRESH:
			return None

		denominator = ((size_r)**2 + (width_r)**2 + (height_r)**2)

		sim = 0
		if denominator != 0:
			sim = 3/denominator
		else:
			sim = np.Inf

		return sim

	else:
		return None


	# make the above the same length as left_cands

if __name__ == "__main__":
	## -- Test setup -- ##
	import stereo_calibration as s_cal
	import funcs as func
	import os

	ball_candidate_dict = {'left': {}, 'right': {}}
	for i in range(1,11):
		ball_candidate_dict['left'][i-1] = np.float32([[i*2,i*2,i*12,i*40,i*40]])
		ball_candidate_dict['right'][i-1] = np.float32([[i*2,i*2,i*12,i*40-i*15,i*40]])

	del ball_candidate_dict['left'][5]
	del ball_candidate_dict['right'][5]
	del ball_candidate_dict['left'][8]
	del ball_candidate_dict['right'][2]

	stereo_calib = s_cal.StereoCal()
	root_p = os.getcwd()
	os.chdir(func.make_path(root_p, c.DATA_DIR, c.STEREO_CALIB_DIR))
	stereo_calib.load_params(c.ACTIVE_STEREO_F)
	os.chdir(root_p)

	## -- ## -- ##

	points_3d = triangulate_points(ball_candidate_dict, stereo_calib)
	for point in points_3d:
		print(point)