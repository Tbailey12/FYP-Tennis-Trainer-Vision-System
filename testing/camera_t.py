import cv2
import os
import time
import multiprocessing as mp
import queue
import numpy as np
import argparse

TEST_F = 'stereo_tests'
IMG_F = 'img'
DATA_F = 'data'
LEFT_F = 'left'
RIGHT_F = 'right'
OUT_F = 'out'
ACTIVE_TEST_F = '2020-03-12_outside_shot_2'
IMG_MEAN_F = 'img_mean.npy'
IMG_STD_F = 'img_std.npy'

ROOT_P = os.getcwd()
TEST_P = ROOT_P + '//' + TEST_F

RESOLUTION = (640,480)
w,h = RESOLUTION
NUM_PROCESSORS = 4

def process_img(img_f, img_queue):
	os.chdir(TEST_P + '//' + img_f + '//' + DATA_F + '//' + ACTIVE_TEST_F)
	
	A =  np.zeros([h,w],dtype=np.uint8)
	B = np.zeros([h,w],dtype=np.uint8)
	B_old = np.zeros([h,w],dtype=np.uint8)
	C = np.zeros([h,w],dtype=np.uint8)

	B_1_std = np.zeros([h,w],dtype=np.float32)
	B_1_mean = np.zeros([h,w],dtype=np.float32)
	B_greater = np.zeros([h,w],dtype=np.uint8)
	B_2_mean = np.zeros([h,w],dtype=np.float32)
	B_less = np.zeros([h,w],dtype=np.uint8)

	img_mean = np.load(IMG_MEAN_F)
	img_std = np.load(IMG_STD_F)

	while True:
		try:
			frame, y_data = img_queue.get_nowait()
			if y_data is None:
				print('done')
				img_queue.task_done()
				return
			## -- Process Image -- ##
			B_old = np.copy(B)
			# B = np.logical_or((y_data > (img_mean + 2*img_std)),
			# (y_data < (img_mean - 2*img_std)))  # foreground new
			np.multiply(img_std,2,out=B_1_std)
			np.add(B_1_std,img_mean,out=B_1_mean)
			B_greater = np.greater(y_data,B_1_mean)
			np.subtract(img_mean,B_1_std,out=B_2_mean)
			B_less = np.less(y_data,B_2_mean)
			B = np.logical_or(B_greater,B_less)

			A = np.invert(np.logical_and(B_old, B))  # difference between prev foreground and new foreground
			C = np.logical_and(A, B)   # different from previous frame and part of new frame
			C = 255*C.astype(np.uint8)

			os.chdir(TEST_P + '//' + img_f + '//' + OUT_F)
			cv2.imwrite(f"{frame}.png", C)

			img_queue.task_done()
		except queue.Empty:
			pass

def read_img(cam_f, img_queue):
	os.chdir(TEST_P + '//' + cam_f + '//' + IMG_F + '//' + ACTIVE_TEST_F)
	img_list = os.listdir()
	
	for img_name in img_list:
		frame = img_name[:4]
		img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
		img_queue.put((frame, img))
	
	for i in range(NUM_PROCESSORS):
		img_queue.put((None, None))
	
	return

if __name__ == "__main__":
	process_list = []
	img_queue = mp.JoinableQueue()

	read_img_proc = mp.Process(target=read_img, args=(LEFT_F, img_queue))
	read_img_proc.start()

	for i in range(NUM_PROCESSORS):
		proc = mp.Process(target=process_img, args=(LEFT_F, img_queue))
		proc.start()
		process_list.append(proc)

	read_img_proc.join()
	img_queue.join()

	# make video from image dif
	os.chdir(TEST_P + '//' + LEFT_F + '//' + OUT_F)
	# Construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
	ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
	args = vars(ap.parse_args())

	output = args['output']
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(output, fourcc, 30, RESOLUTION)

	img_list = os.listdir()
	for img in img_list:
		if '.png' in img:
			out.write(cv2.imread(img))

	out.release()
