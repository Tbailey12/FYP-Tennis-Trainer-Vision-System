import numpy as np 	
import consts as c
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

if __name__ == "__main__":
	points = np.load('points_3d.npy', allow_pickle=True)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('x (m)')
	ax.set_ylabel('y (m)')
	ax.set_zlabel('z (m)')
	ax.set_xlim(-1.5, 1.5)
	ax.set_ylim(0, 3)
	ax.set_zlim(0, 3)

	for frame, point in enumerate(points):
		if len(point) > 0 and point.item(2)>0 and point.item(1)<0:
			ax.scatter(xs=point.item(0)-(25E-2/2),ys=point.item(2),zs=-point.item(1))
	plt.savefig(f"{frame:04d}.png")
		# plt.show()