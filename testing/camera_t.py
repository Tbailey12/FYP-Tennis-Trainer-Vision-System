import cv2
import os
import time
import multiprocessing as mp
import queue
import numpy as np

TEST_F = 'stereo_tests'
IMG_F = 'img'
DATA_F = 'data'
RIGHT_F = 'left'
RIGHT_F = 'right'
OUT_F = 'out'
# ACTIVE_TEST_F = '2020-03-12_outside_shot_2'
IMG_MEAN_F = 'img_mean.npy'
IMG_STD_F = 'img_std.npy'

ROOT_P = os.getcwd()
TEST_P = ROOT_P + '//' + TEST_F

RESOLUTION = (640,480)
w,h = RESOLUTION
NUM_PROCESSORS = 4
p = 0.01

SIZE = 2
X = 3
Y = 4

kernel = np.array([	[0,1,0],
					[1,1,1],
					[0,1,0]], dtype=np.uint8)

kernel1 = np.array([[0,0,1,0,0],
					[0,1,1,1,0],
					[1,1,1,1,1],
					[0,1,1,1,0],
					[0,0,1,0,0]], dtype=np.uint8)

def process_img(img_f, img_queue, active_test_dir, out_q):
	os.chdir(TEST_P + '//' + img_f + '//' + DATA_F + '//' + active_test_dir)
	
	A =  np.zeros([h,w],dtype=np.uint8)
	B = np.zeros([h,w],dtype=np.uint8)
	B_old = np.zeros([h,w],dtype=np.uint8)
	C = np.zeros([h,w],dtype=np.uint8)

	B_1_std = np.zeros([h,w],dtype=np.float32)
	B_1_mean = np.zeros([h,w],dtype=np.float32)
	B_greater = np.zeros([h,w],dtype=np.uint8)
	B_2_mean = np.zeros([h,w],dtype=np.float32)
	B_less = np.zeros([h,w],dtype=np.uint8)

	mean_1 = np.zeros([h,w],dtype=np.float32)
	mean_2 = np.zeros([h,w],dtype=np.float32)

	std_1 = np.zeros([h,w],dtype=np.float32)
	std_2 = np.zeros([h,w],dtype=np.float32)
	std_3 = np.zeros([h,w],dtype=np.float32)
	std_4 = np.zeros([h,w],dtype=np.float32)
	std_5 = np.zeros([h,w],dtype=np.float32)
	std_6 = np.zeros([h,w],dtype=np.float32)

	img_mean = np.load(IMG_MEAN_F)
	img_std = np.load(IMG_STD_F)

	while True:
		try:
			frame, y_data = img_queue.get_nowait()
			if y_data is None:
				img_queue.task_done()
				return

			## -- Process Image -- ##
			B_old = np.copy(B)
			# B = np.logical_or((y_data > (img_mean + 2*img_std)),
			# (y_data < (img_mean - 2*img_std)))  # foreground new
			np.multiply(img_std,3,out=B_1_std)
			np.add(B_1_std,img_mean,out=B_1_mean)
			B_greater = np.greater(y_data,B_1_mean)
			np.subtract(img_mean,B_1_std,out=B_2_mean)
			B_less = np.less(y_data,B_2_mean)
			B = np.logical_or(B_greater,B_less)

			C = B
			C = 255*C.astype(np.uint8)

			C = cv2.morphologyEx(C, cv2.MORPH_CLOSE, kernel, iterations=1)
			C = cv2.erode(C, kernel1, iterations=1)

			# for i in ec_order:
			# 	if i == 'e':
			# 		C = cv2.erode(C, k1, iterations=1)
			# 	elif i == 'c':
			# 		C = cv2.morphologyEx(C, cv2.MORPH_CLOSE, k2, iterations=1)

			# calculate new std deviation and mean
			y_data = np.bitwise_and(y_data, np.invert(C))
			# img_mean = (1 - p) * img_mean + p * y_data  # calculate mean
			np.multiply(1-p,img_mean,out=mean_1)
			np.multiply(p,y_data,out=mean_2)
			np.add(mean_1,mean_2,out=img_mean)
			
			# img_std = np.sqrt((1 - p) * (img_std ** 2) + p * ((y_data - img_mean) ** 2))  # calculate std deviation
			np.square(img_std,out=std_1)
			np.multiply(1-p,std_1,out=std_2)
			np.subtract(y_data,img_mean,out=std_3)
			np.square(std_3,out=std_4)
			np.multiply(p,std_4,out=std_5)
			np.add(std_2,std_5,out=std_6)
			np.sqrt(std_6,out=img_std)

			## -- object detection -- ##
			n_features_cv, labels_cv, stats_cv, centroids_cv = cv2.connectedComponentsWithStats(C, connectivity=4)

			label_mask_cv = np.logical_and(stats_cv[:,cv2.CC_STAT_AREA]>0, stats_cv[:,cv2.CC_STAT_AREA]<50000)
			ball_candidates = np.concatenate((stats_cv[label_mask_cv,2:],centroids_cv[label_mask_cv]), axis=1)

			out_q.put((frame, ball_candidates))

			os.chdir(TEST_P + '//' + img_f + '//' + IMG_F + '//' + active_test_dir)
			img = cv2.imread(f"{frame}.png")

			C = cv2.cvtColor(C, cv2.COLOR_GRAY2RGB)

			ball_candidates = ball_candidates.astype(int)
			for ball in ball_candidates:
				cv2.drawMarker(C,(ball[X],ball[Y]),(0, 0, 255),cv2.MARKER_CROSS,thickness=2,markerSize=10)
				cv2.drawMarker(img,(ball[X],ball[Y]),(0, 0, 255),cv2.MARKER_CROSS,thickness=2,markerSize=10)

			os.chdir(TEST_P + '//' + img_f + '//' + OUT_F + '//' + active_test_dir)
			cv2.imwrite(f"C{frame}.png", C)
			cv2.imwrite(f"{frame}.png", img)

			img_queue.task_done()
		except queue.Empty:
			pass

def read_img(cam_f, queue_list, active_test_dir):
	os.chdir(TEST_P + '//' + cam_f + '//' + IMG_F + '//' + active_test_dir)
	img_list = os.listdir()
	
	cur_queue = 0
	for img_name in img_list:
		frame = img_name[:4]
		img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
		queue_list[cur_queue].put((frame, img))
		cur_queue += 1

		if cur_queue >= NUM_PROCESSORS:
			cur_queue = 0
	
	for i in range(NUM_PROCESSORS):
		queue_list[i].put((None, None))
	
	return

if __name__ == "__main__":
	ball_candidates_out = []

	# get list of test data directories
	os.chdir(TEST_P + '//' + RIGHT_F + '//' + DATA_F)
	test_directories = os.listdir()
	os.chdir(TEST_P + '//' + RIGHT_F + '//' + OUT_F)
	for directory in test_directories:
		try:
			os.mkdir(directory)
		except FileExistsError:
			pass

	# directory = test_directories[4]
	# print(directory)
	for directory in test_directories[4:5]:
		active_test_dir = str(directory)
		os.chdir(ROOT_P)
		process_list = []
		queue_list = []
		out_q = mp.JoinableQueue()
 
		for i in range(NUM_PROCESSORS):
			q = mp.JoinableQueue()
			queue_list.append(q)
			proc = mp.Process(target=process_img, args=(RIGHT_F, q, active_test_dir, out_q))
			proc.start()
			process_list.append(proc)

		read_img_proc = mp.Process(target=read_img, args=(RIGHT_F, queue_list, active_test_dir))
		read_img_proc.start()


		read_img_proc.join()
		for q in queue_list:
			q.join()

		while True:
			try:
				candidate = out_q.get_nowait()
				ball_candidates_out.append(candidate)

			except queue.Empty:
				break

		os.chdir(TEST_P + '//' + RIGHT_F + '//' + DATA_F)
		np.save('right_ball_candidates.npy', ball_candidates_out)

		# make video from image dif
		img_list = []
		C_list = []

		os.chdir(TEST_P + '//' + RIGHT_F + '//' + OUT_F + '//' + active_test_dir)
		im_names = os.listdir()
		for im in im_names:
			if 'C' in im:
				C_list.append(cv2.imread(im))
			else:
				img_list.append(cv2.imread(im))

		# os.chdir(TEST_P + '//' + RIGHT_F + '//' + OUT_F + '//' + active_test_dir)
		# im_names = os.listdir()
		# for im in im_names:
		# 	C_list.append(cv2.imread(im))

		os.chdir(TEST_P + '//' + RIGHT_F + '//' + OUT_F)
		
		output = f"{active_test_dir}.mp4"
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		out = cv2.VideoWriter(output, fourcc, 30, (w,h*2))

		for i, img in enumerate(img_list):
			out_img = cv2.vconcat([img,C_list[i]])
			out.write(out_img)

		out.release()
		print(f"{active_test_dir} done")