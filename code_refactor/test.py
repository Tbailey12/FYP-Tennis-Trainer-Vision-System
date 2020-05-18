import numpy as np 	
import consts as c
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os

import consts as c
import triangulation as tr
import funcs as func
import stereo_calibration as s_cal
import server as serv

root_p = os.getcwd()

def load_stereo_calib(filename):
        stereo_calib = s_cal.StereoCal()
        try:
            os.chdir(func.make_path(root_p, c.DATA_DIR, c.STEREO_CALIB_DIR))
            stereo_calib.load_params(filename)
            os.chdir(root_p)
            return stereo_calib

        except ValueError as e:
            print(e)
            return None

if __name__ == "__main__":
	stereo_calib = load_stereo_calib(c.ACTIVE_STEREO_F)
	with open("ball_dict.pkl", "rb") as file:
		ball_candidate_dict = pickle.load(file)

	points_3d = tr.triangulate_points(ball_candidate_dict, stereo_calib)
	trajectory = serv.analyse_trajectory(points_3d)

	serv.plot_trajectory(trajectory)

	quit()
	points_3d = np.load('points_3d.npy', allow_pickle=True)


	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('x (m)')
	ax.set_ylabel('y (m)')
	ax.set_zlabel('z (m)')
	ax.set_xlim(-1.5, 1.5)
	ax.set_ylim(0, 2.4)
	ax.set_zlim(0, 0.3)

	# for tok in trajectory.tokens:
	# 	ax.scatter(xs=tok.coords[0],ys=tok.coords[1],zs=tok.coords[2])

	for frame in points_3d:
		for point in frame:
			if len(point) > 0:
				print(point)
				ax.scatter(xs=point[0],ys=point[1],zs=point[2])
	# plt.savefig(f"{frame:04d}.png")
	plt.show()