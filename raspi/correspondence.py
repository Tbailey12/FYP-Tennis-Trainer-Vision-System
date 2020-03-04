import numpy as np 

SIZE = 4
X = 5
Y = 6

if __name__ == "__main__":
	left_balls = np.load('left_ball_candidates.npy', allow_pickle=True)
	right_balls = np.load('right_ball_candidates.npy', allow_pickle=True)

	left_balls = left_balls[np.argsort(left_balls[:,0])]
	right_balls = right_balls[np.argsort(right_balls[:,0])]

	print(len(left_balls[79,1].astype(int)))
	# for i in range(min(len(left_balls),len(right_balls))):
		# if len(left_balls[i,1]) > 0 and len(right_balls[i,1] > 0):
		# 	print(left_balls[i,1][1])
			# print(right_balls[i,1])

			### Rectify all feature points
			### calculate similarity score between points in L/R arrays