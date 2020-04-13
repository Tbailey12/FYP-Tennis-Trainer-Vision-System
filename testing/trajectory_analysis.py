import numpy as np
import os
import time

ROOT_P = 'D:\\documents\\local uni\\FYP\\code'

X = 0
Y = 1
Z = 2
FPS = 90
VM = 150	# max ball velocity
Vm = 30		# min ball velocity
WIN_SIZE = 20
WIN_OVERLAP = 3
MAX_EST = 3
dT = 1/FPS	# inter frame time

def kph_2_mps(kph):
	return kph*10/36

dM = kph_2_mps(VM)*dT 	# max dist
thetaM = np.pi
phiM = np.pi

class Tracklet(object):
	def __init__(self, start_frame):
		self.start_frame = start_frame
		self.tokens = []
		self.score = 0
		self.length = 0
		self.con_est = 0

	def add_token(self, token):
		self.con_est = 0
		self.tokens.append(token)
		self.score += token.score
		self.length += 1

	def del_token(self):
		self.score -= self.tokens[-1].score
		self.length -= 1
		del self.tokens[-1]

	def est_next(self):
		if self.length >= 3:
			a3 = ((self.tokens[-1].coords-self.tokens[-2].coords)-(self.tokens[-2].coords-self.tokens[-3].coords))/(dT**2)			# acceleration
			v3 = ((self.tokens[-1].coords-self.tokens[-2].coords)/(dT)) + a3*dT 			# velocity

			c4_e = self.tokens[-1].coords+v3*dT+((a3*dT**2)/(2)) 		# next point estimation
			return c4_e
		else:
			return None

	def add_est(self, token):
		con_est_temp = 0
		if self.con_est < MAX_EST:
			con_est_temp += self.con_est+1
			self.add_token(token)
			self.con_est = con_est_temp
		
			return True
		else:
			print('too many')
			return False

class Token(object):
	def __init__(self, f, coords, score=0):
		self.f = f 
		self.coords = coords
		self.score = score

def calc_dist(vect):
	a = 0
	for el in vect:
		a += el**2

	return np.sqrt(a)

def calc_theta_phi(diff, r):
	r_p = np.sqrt(diff[X]**2+diff[Y]**2)

	theta = np.arccos(diff[Y]/r_p)
	phi = np.arccos(r_p/r)
	return theta, phi

def score_node(est, candidate):
	diff = est-candidate
	r = calc_dist(diff)
	theta, phi = calc_theta_phi(diff, r)

	if r<dM:
		s1 = np.exp(-r/dM)
		s2 = np.exp(-theta/thetaM)
		s3 = np.exp(-phi/phiM)
		return s1+s2+s3
	else:
		return 0

def evaluate(candidates_3D, tracklet, f, f_max):
	done = False
	if f < f_max:
		est = tracklet.est_next()
		if candidates_3D[f] == []:
			print('estimate')
			if not tracklet.add_est(Token(f, est)):
				done = True
		else:
			for c4 in candidates_3D[f]:
				score = score_node(est, c4)
				tracklet.add_token(Token(f, c4, score))
				if evaluate(candidates_3D, tracklet, f+1, f_max):
					done = True
	if done:
		print('hi', f)
	else:
		return True

if __name__ == "__main__":
	RESOLUTION = (640,480)
	w,h = RESOLUTION

	os.chdir(ROOT_P + '\\' + 'img\\inside_tests')

	candidates_3D = np.load('candidates_3D.npy', allow_pickle=True)

	for i in range(108,len(candidates_3D)):
		candidates_3D[i] = []

	## -- Shift Token Transfer -- ##
	frame_num = len(candidates_3D)
	num_windows = int(np.floor(frame_num/WIN_SIZE))

	windows = num_windows*[None]

	for window in range(0,num_windows*WIN_SIZE,(WIN_SIZE-WIN_OVERLAP)):
		init_set = False
		c1,c2,c3,c4 = [],[],[],[]
	
		if (frame_num-window) < WIN_SIZE:
			break

		for f in range(window+3, window+WIN_SIZE):
			if init_set is False:
				c1 = candidates_3D[f-3]
				c2 = candidates_3D[f-2]
				c3 = candidates_3D[f-1]

				if (c1 == []) or (c2 == []) or (c3 == []):
					continue
				else:
					init_set = True

			tracklet = Tracklet(f)
			for c1_c in c1:
				tracklet.add_token(Token(f-3,c1_c))
				for c2_c in c2:
					tracklet.add_token(Token(f-2,c2_c))
					for c3_c in c3:
						tracklet.add_token(Token(f-1,c3_c))

						evaluate(candidates_3D, tracklet, f, f_max=(window+WIN_SIZE))				
						print(tracklet.length)
	print(frame_num)








































	# ## -- Plot points -- ##
	# import matplotlib.pyplot as plt
	# from mpl_toolkits.mplot3d import Axes3D
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# ax.set_xlabel('x (m)')
	# ax.set_ylabel('y (m)')
	# ax.set_zlabel('z (m)')
	# ax.set_xlim(-11E-1/2, 11E-1/2)
	# ax.set_ylim(0, 24E-1)
	# ax.set_zlim(0, 2E-1)

	# os.chdir("plots")

	# count = 0
	# for f, candidates in enumerate(candidates_3D):
	# 	for candidate in candidates:
	# 		ax.scatter(xs=candidate[0],ys=candidate[1],zs=candidate[2])
	# 		count+=1
	# 	# plt.savefig(f"{f:04d}.png")


	# print(count)
	# plt.show()
	# quit()