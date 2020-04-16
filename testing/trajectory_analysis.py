import numpy as np
import os
import time
import copy

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
C_INIT = 0
CAND = 1
TRACKLET_SCORE_THRESH = 1
TOKEN_SIM_THRESH = 0
TOKEN_SCORE_THRESH = 0
SCORE_TOK_THRESH = 0

def kph_2_mps(kph):
	return kph*10/36

dM = kph_2_mps(VM)*dT*2 	# max dist
thetaM = np.pi
phiM = np.pi

class TrackletBox(object):
	def __init__(self):
		self.tracklets = []

	def add_tracklet(self, tracklet):
		self.tracklets.append(tracklet)

	def validate_tracklets(self):
		for tracklet in self.tracklets:
			score_tok = tracklet.score/tracklet.length
			if score_tok < SCORE_TOK_THRESH:
				tracklet.is_valid = False

	def merge_tracklets(self):
		for t1 in self.tracklets:
			hiscore = t1.score
			for t2 in self.tracklets:
				if t1 is not t2:
					if t1.start_frame == t2.start_frame:
						# tracklets start at the same point
						# remove the tracklet with the lower score
						if t2.score > hiscore:
							hiscore = t2.score
							t1.is_valid = False
						else:
							t2.is_valid = False

		for t1 in self.tracklets:
			for t2 in self.tracklets:
				if t1 is not t2 and t1.is_valid and t2.is_valid:
					# check for temporal overlap
					first = None
					second = None
					if t2.start_frame > t1.start_frame and t2.start_frame <= t1.start_frame+t1.length:
						# t2 starts inside t1
						first = t1
						second = t2
					elif t1.start_frame > t2.start_frame and t1.start_frame <= t2.start_frame+t2.length:
						# t1 starts inside t2
						first = t2
						second = t1

					if first is not None and second is not None:
						# if temporal overlap
						contained = None
						if second.start_frame+second.length < first.start_frame+first.length:
							contained = True
						else:
							contained = False
						# check spatial overlap
						if contained:
							pass
						else:
							shared_tracklets = []
							for token1 in reversed(first.tokens):
								cons = False
								cons_count = 0
								# track = Tracklet(first.tokens.index(token1))
								for token2 in second.tokens:
									sim = token1.calc_similarity(token2)

									if sim < TOKEN_SIM_THRESH:
										cons = True
										cons_count += 1
										first_index = first.tokens.index(token1)
										second_index = second.tokens.index(token2)
									else:
										if cons is True:
											shared_tracklets.append([first_index, second_index, cons_count])
											break
							# find the track with the most shared tokens
							if shared_tracklets != []:
								shared_track = sorted(shared_tracklets, key=lambda x: x[2], reverse=True)[0]
								
								first.tokens = first.tokens[0:shared_track[0]+1]
								for tok in second.tokens[shared_track[1]:]:
									first.add_token(tok)

								second.is_valid = False

							# if num_shared > 0:
							# 	first.tokens
							# if spatial overlap
								# merge tracklets
							# if not spatial overlap
								# don't merge tracklets
					# if not temporal overlap
						# check spatial distance
							# if distance is < D
								# extrapolate ends of both tracklets
								# if they intersect
									# extend tracklets to intersection point
								# else don't extend them


class Tracklet(object):
	def __init__(self, start_frame, tracklet_box=None, tokens=[], score=0,length=0):
		self.start_frame = start_frame
		self.tracklet_box = tracklet_box
		self.tokens = tokens
		self.score = score
		self.length = length
		self.con_est = 0
		self.is_valid = True

	def save_tracklet(self):
		if self.score < TRACKLET_SCORE_THRESH:
			return
		if tracklet_box is not None:
			self.tracklet_box.add_tracklet(Tracklet(start_frame = copy.deepcopy(self.start_frame),\
						tokens = copy.deepcopy(self.tokens), score = copy.deepcopy(self.score), \
						length = copy.deepcopy(self.length)))
			# self.tracklet_box.add_tracklet(copy.deepcopy(self))


	def add_token(self, token):
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
		if self.con_est < MAX_EST:
			self.add_token(token)
			self.con_est += 1
			return True
		else:
			self.con_est = 0
			return False

class Token(object):
	def __init__(self, f, coords, score=0):
		self.f = f 
		self.coords = coords
		self.score = score
		self.is_valid = True

	def calc_similarity(self, token):
		error = self.coords-token.coords
		e_sum = 0
		for c in error:
			e_sum+=c**2
		return np.sqrt(e_sum)

def calc_dist(vect):
	a = 0
	for el in vect[:2]:
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
	# want to stop recursion when
	# - 3 consecutive estimates are made
	# - f == max
	if f < f_max:
		est = tracklet.est_next()

		# are there tokens to compare?
		# yes: compare the tokens and continue
		# no: add the estimate as a token and continue
		if candidates_3D[f] != []:
			for i, cand in enumerate(candidates_3D[f]):
				c4 = cand[CAND]
				candidates_3D[f][i][C_INIT] = True
				score = score_node(est, c4)
				if score > TOKEN_SCORE_THRESH:
					tracklet.add_token(Token(f, c4, score))
					evaluate(candidates_3D, tracklet, f+1, f_max)
					tracklet.del_token()
		else:
			if tracklet.add_est(Token(f, est)):
				evaluate(candidates_3D, tracklet, f+1, f_max)
				tracklet.del_token()
			else:
				# added 3 estimates, stop recursion and save tracklet[-3]
				tracklet.save_tracklet()
	else:
		# tracklet is max length
		# save tracklet and stop recursion
		tracklet.save_tracklet()

def check_init_toks(c1,c2,c3):
	d1 = calc_dist(c2-c1)
	d2 = calc_dist(c3-c2)

	if d1<dM and d2<dM and d1>0 and d2>0:
		return True
	else:
		return False

if __name__ == "__main__":
	RESOLUTION = (640,480)
	w,h = RESOLUTION

	os.chdir(ROOT_P + '\\' + 'img\\simulation_tests')

	candidates_3D = np.load('candidates_3D.npy', allow_pickle=True)

	for f, frame in enumerate(candidates_3D):
		for c, candidate in enumerate(frame):
			candidates_3D[f][c] = [False, np.array(candidate)]

	## -- Shift Token Transfer -- ##
	frame_num = len(candidates_3D)
	num_windows = int(np.floor(frame_num/WIN_SIZE))

	windows = num_windows*[None]

	tracklet_box = TrackletBox()
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

			tracklet = Tracklet(f, tracklet_box)
			for c1_c in c1:
				if c1_c[C_INIT] is True:	continue
				tracklet.add_token(Token(f-3,c1_c[CAND], score=1))
				for c2_c in c2:
					if c2_c[C_INIT] is True:	continue
					tracklet.add_token(Token(f-2,c2_c[CAND], score=1))
					for c3_c in c3:
						if c3_c[C_INIT] is True:	continue
						tracklet.add_token(Token(f-1,c3_c[CAND], score=1))

						if check_init_toks(c1_c[CAND],c2_c[CAND],c3_c[CAND]):
							evaluate(candidates_3D, tracklet, f, f_max=(window+WIN_SIZE))				

						tracklet.del_token()
					tracklet.del_token()
				tracklet.del_token()

			init_set = False
			c1,c2,c3,c4 = [],[],[],[]

	tracklet_box.merge_tracklets()
	tracklet_box.validate_tracklets()

	## -- Plot points -- ##
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('x (m)')
	ax.set_ylabel('y (m)')
	ax.set_zlabel('z (m)')
	ax.set_xlim(-11/2, 11/2)
	ax.set_ylim(0, 24)
	ax.set_zlim(-2, 2)

	# count = 0
	# for f, candidates in enumerate(candidates_3D):
	# 	for candidate in candidates:
	# 		ax.scatter(xs=candidate[0],ys=candidate[1],zs=candidate[2])
	# 		count+=1
	# 	# plt.savefig(f"{f:04d}.png")


	# print(count)
	# plt.show()
	# quit()


	count = 0
	for tracklet in tracklet_box.tracklets:
		if tracklet.is_valid:
			for tok in tracklet.tokens:
				ax.scatter(xs=tok.coords[X], ys=tok.coords[Y], zs=tok.coords[Z])
	plt.show()

	print(f"Tracklet number: {len(tracklet_box.tracklets)}")
	count = 0
	for t in tracklet_box.tracklets:
		if t.is_valid:
			count += 1
			print(f"f_start: {t.start_frame}, f_end: {t.start_frame+t.length}, score: {t.score:0.2f}, score/tok: {t.score/t.length:0.2f}")
	






















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