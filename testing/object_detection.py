import os
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import stereo_calibration_t as s_cal

l_dir = 'left'
r_dir = 'right'
out_dir = 'out'

target_dir = 'no_filter_C_outside_2'

kern = np.array([	[0,1],
					[1,1]], dtype=np.uint8)

kernel = np.array([	[0,1,0],
					[1,1,1],
					[0,1,0]], dtype=np.uint8)

kernel1 = np.array([[0,0,1,0,0],
					[0,1,1,1,0],
					[1,1,1,1,1],
					[0,1,1,1,0],
					[0,0,1,0,0]], dtype=np.uint8)

kernel2 = np.array([[0,0,1,1,0,0],
					[0,1,1,1,1,0],
					[1,1,1,1,1,1],
					[1,1,1,1,1,1],
					[0,1,1,1,1,0],
					[0,0,1,1,0,0]], dtype=np.uint8)

kernel3 = np.array([[0,0,1,1,1,1,0,0],
					[0,1,1,1,1,1,1,0],
					[1,1,1,1,1,1,1,1],
					[1,1,1,1,1,1,1,1],
					[1,1,1,1,1,1,1,1],
					[0,1,1,1,1,1,1,0],
					[0,0,1,1,1,1,0,0]], dtype=np.uint8)

SIZE = 2
X = 3
Y = 4
N_OBJECTS = 10

def find_balls(root_dir, img_dir, out_img_list, ball_list, ROI):
	os.chdir(root)
	os.chdir(target_dir+'//'+img_dir)
	img_list = glob("****.png")
	C_img_list = glob("C****.png")


	ball_candidates = []

	for i, C_img in enumerate(C_img_list):
		## -- binary filtering -- ##
		img = cv2.imread(C_img,cv2.IMREAD_GRAYSCALE)
		C = np.zeros(img.shape, dtype=np.uint8)
		C[ROI[1]:ROI[3],ROI[0]:ROI[2]] = img[ROI[1]:ROI[3],ROI[0]:ROI[2]]

		C = cv2.erode(C, kernel, iterations=1)
		C = cv2.morphologyEx(C, cv2.MORPH_CLOSE, kernel2, iterations=1)
		
		## -- object detection -- ##
		n_features_cv, labels_cv, stats_cv, centroids_cv = cv2.connectedComponentsWithStats(C, connectivity=4)

		label_mask_cv = np.logical_and(stats_cv[:,cv2.CC_STAT_AREA]>5, stats_cv[:,cv2.CC_STAT_AREA]<50000)
		ball_candidates = np.concatenate((stats_cv[label_mask_cv,2:],centroids_cv[label_mask_cv]), axis=1)
		
		# sort ball candidates by size and keep the top 10
		ball_candidates = ball_candidates[ball_candidates[:,SIZE].argsort()[::-1][:N_OBJECTS]]

		ball_list.append((i, ball_candidates))
		ball_candidates = ball_candidates.astype(int)

		## -- adding circles -- ##
		C = cv2.cvtColor(C, cv2.COLOR_GRAY2RGB)
		img = cv2.imread(img_list[i])
		
		for ball in ball_candidates:
			if ball[SIZE] > 81:
				cv2.drawMarker(C,(ball[X],ball[Y]),(0, 0, 255),cv2.MARKER_CROSS,thickness=2,markerSize=10)
				cv2.drawMarker(img,(ball[X],ball[Y]),(0, 0, 255),cv2.MARKER_CROSS,thickness=2,markerSize=10)

		img2 = cv2.vconcat([img,C])
		out_img_list.append(img2)
	os.chdir(root)

if __name__ == "__main__":
	root = os.getcwd()

	left_imgs = []
	right_imgs = []

	left_balls = []
	right_balls = []

	s_calib = s_cal.StereoCal()
	s_calib = s_calib.load_params("stereo_calib.npy")

	find_balls(root, l_dir, left_imgs, left_balls, s_calib.validPixROI1)
	find_balls(root, r_dir, right_imgs, right_balls, s_calib.validPixROI2)

	os.chdir(root)
	np.save('left_ball_candidates.npy', left_balls)
	np.save('right_ball_candidates.npy', right_balls)


	os.chdir(target_dir+'//'+out_dir)

	# list_len = min(len(left_imgs),len(right_imgs))
	# for i in range(list_len):
	# 	if i>3:
	# 		img_out = cv2.hconcat([left_imgs[i],right_imgs[i]])
	# 		cv2.imshow('img',img_out)
	# 		cv2.waitKey(30)
	# 		cv2.imwrite(f"{i:04d}.png",img_out)


	# cv.hconcat()
