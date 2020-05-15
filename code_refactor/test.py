import numpy as np 	
import consts as c
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

if __name__ == "__main__":
	points_3d = np.load('points_3d.npy', allow_pickle=True)


	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('x (m)')
	ax.set_ylabel('y (m)')
	ax.set_zlabel('z (m)')
	ax.set_xlim(-1.5, 1.5)
	ax.set_ylim(0, 2.4)
	ax.set_zlim(0, 0.3)

	for frame in points_3d:
		for point in frame:
			if len(point) > 0:
				print(point)
				ax.scatter(xs=point[0],ys=point[1],zs=point[2])
	# plt.savefig(f"{frame:04d}.png")
	plt.show()