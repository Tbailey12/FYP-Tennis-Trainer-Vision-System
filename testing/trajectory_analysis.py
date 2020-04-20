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
WIN_SIZE = 40
WIN_OVERLAP = 0
MAX_EST = 3
dT = 1/FPS	# inter frame time
C_INIT = 0
CAND = 1
EXTRAPOLATE_N = 3


def kph_2_mps(kph):
	return kph*10/36

dM = kph_2_mps(VM)*dT 	# max dist
thetaM = np.pi
phiM = np.pi

TRACKLET_SCORE_THRESH = 1
TOKEN_SIM_THRESH = dM/2
TOKEN_SCORE_THRESH = 1
SCORE_TOK_THRESH = 1

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

	def split_tracklets(self):
		new_tracklets = []
		for t in self.tracklets:
			acc = []
			vel = []
			if t.is_valid:
				for i, tok in enumerate(t.tokens):
					if i==0:
						vel.append(0*tok.coords)
					else:
						vel.append(t.tokens[i].coords-t.tokens[i-1].coords)

				for j, v in enumerate(vel):
					if j<3:
						acc.append(0)
					else:
						if vel[j][Z] > 0 and vel[j-1][Z] < 0 and vel[j-2][Z] < 0 and vel[j-3][Z] < 0:
							acc.append(1)
						else: 
							acc.append(-1)

				split_start_f = 0
				for k, a in enumerate(acc):
					if k<2 or k>=len(acc)-1:
						pass
					else:
						if acc[k] > 0 and acc[k-1] <= 0 and acc[k+1] <=0 :
							new_track = Tracklet(split_start_f, \
							tokens = t.tokens[split_start_f:k], \
							score = self.tok_score_sum(t.tokens[split_start_f:k]), \
							length = len(t.tokens[split_start_f:k]))

							t.is_valid = False
							self.tracklets.append(new_track)
							split_start_f = k



	def merge_tracklets(self):
		## -- Same start frame -- ##
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

		# -- Temporal overlap -- ##
		for t1 in self.tracklets:
			if not t1.is_valid:
				continue

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
								first.length = len(first.tokens)

								for tok in second.tokens[shared_track[1]:]:
									first.add_token(tok)

								second.is_valid = False

		# tracklets intersect after extrapolation
		for t1 in self.tracklets:
			if not t1.is_valid:
				continue
			for t2 in self.tracklets:
				if not t2.is_valid:
					continue
				first = None
				second = None
				if t1.start_frame+t1.length < t2.start_frame:
					first = t1
					second = t2
				elif t2.start_frame+t2.length < t1.start_frame:
					first = t2
					second = t1


				if first is not None and second is not None:
					if first.length > 3 and second.length > 3:
						first_extrapolation_points = []
						second_extrapolation_points = []

						for i in range(3):
							first_extrapolation_points.append(first.tokens[i-3].coords)
							second_extrapolation_points.append(second.tokens[2-i].coords)

						for i in range(EXTRAPOLATE_N):
							first_extrapolation_points.append(
								make_est(	first_extrapolation_points[-3],
											first_extrapolation_points[-2],
											first_extrapolation_points[-1]))
							
							second_extrapolation_points.append(
								make_est(	second_extrapolation_points[-3],
											second_extrapolation_points[-2],
											second_extrapolation_points[-1]))

						first_extrapolation_points = first_extrapolation_points[-EXTRAPOLATE_N:]
						second_extrapolation_points = second_extrapolation_points[-EXTRAPOLATE_N:]

						best_match = TOKEN_SIM_THRESH
						best_f_p = None
						best_s_p = None
						for i, f_p in enumerate(first_extrapolation_points):
							for j, s_p in enumerate(second_extrapolation_points):
								sim = calc_dist(f_p-s_p)
								if sim < TOKEN_SIM_THRESH:
									best_match = sim
									best_f_p = i
									best_s_p = j
									break
							if best_f_p is not None:
								break

						if best_f_p is not None and best_s_p is not None:
							new_first_points = first_extrapolation_points[:i]
							new_second_points = second_extrapolation_points[:j]

							for first_point in new_first_points:
								first.add_token(Token(first.tokens[-1].f+1,first_point,score=1))

							for second_point in reversed(new_second_points):
								first.add_token(Token(first.tokens[-1].f+1,second_point,score=1))

							for tok in second.tokens:
								first.add_token(tok)

							second.is_valid = False

	def tok_score_sum(self, tokens):
		score = 0
		for tok in tokens:
			score += tok.score

		return score

def make_est(c1,c2,c3):
	a3 = ((c3-c2)-(c2-c1))/(dT**2)			# acceleration
	v3 = ((c3-c2)/(dT)) + a3*dT 			# velocity

	c4_e = c3+v3*dT+((a3*dT**2)/(2)) 		# next point estimation
	return c4_e

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

	def insert_token(self, token, index):
		if index < len(self.tokens):
			self.tokens.insert(index, token)
			self.length+=1
			self.score+=token.score

	def del_token(self):
		self.score -= self.tokens[-1].score
		self.length -= 1
		del self.tokens[-1]

	def est_next(self):
		if self.length >= 3:
			est = make_est(self.tokens[-3].coords,self.tokens[-2].coords,self.tokens[-1].coords)
			return est
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
		return calc_dist(error)

def calc_dist(vect):
	a = 0
	for el in vect[:3]:
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
	# want to stop recursion when
	# - 3 consecutive estimates are made
	# - f == max
	if f < f_max:
		est = tracklet.est_next()

		# are there tokens to compare?
		# yes: compare the tokens and continue
		# no: add the estimate as a token and continue
		valid_cand = False
		if candidates_3D[f] != []:
			valid_cand = False
			for i, cand in enumerate(candidates_3D[f]):
				c4 = cand[CAND]
				# candidates_3D[f][i][C_INIT] = True
				score = score_node(est, c4)
				if score > TOKEN_SCORE_THRESH:
					valid_cand = True
					tracklet.add_token(Token(f, c4, score))
					evaluate(candidates_3D, tracklet, f+1, f_max)
					tracklet.del_token()

		if valid_cand is False:
			if tracklet.add_est(Token(f, est)):
				evaluate(candidates_3D, tracklet, f+1, f_max)
				tracklet.del_token()
			else:
				# added 3 estimates, stop recursion and save tracklet
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

	os.chdir(ROOT_P + '\\' + 'img\\inside_tests')

	candidates_3D = np.load('candidates_3D.npy', allow_pickle=True)


	## -- Plot points -- ##
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('x (m)')
	ax.set_ylabel('y (m)')
	ax.set_zlabel('z (m)')
	ax.set_xlim(-11E-1/2, 11E-1/2)
	ax.set_ylim(0, 24E-1)
	ax.set_zlim(0, 3E-1)

	for f, frame in enumerate(candidates_3D):
		for c, candidate in enumerate(frame):
			candidates_3D[f][c] = [False, np.array(candidate)]

	## -- Shift Token Transfer -- ##
	frame_num = len(candidates_3D)
	num_windows = int(np.floor(frame_num/WIN_SIZE))

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

						c1_c[C_INIT] = True
						c2_c[C_INIT] = True
						c3_c[C_INIT] = True

						if check_init_toks(c1_c[CAND],c2_c[CAND],c3_c[CAND]):
							evaluate(candidates_3D, tracklet, f, f_max=(window+WIN_SIZE))				

						tracklet.del_token()
					tracklet.del_token()
				tracklet.del_token()

			init_set = False
			c1,c2,c3,c4 = [],[],[],[]

	for tracklet in tracklet_box.tracklets:
		for tok in tracklet.tokens:
			ax.scatter(xs=tok.coords[X],ys=tok.coords[Y],zs=tok.coords[Z])
	# 	plt.show()
	# quit()
	tracklet_box.merge_tracklets()
	tracklet_box.validate_tracklets()
	tracklet_box.split_tracklets()

	best_score, best_tracklet = 0, None
	for t in tracklet_box.tracklets:
		if t.is_valid:
			print(f"f_start: {t.start_frame}, f_end: {t.start_frame+t.length}, score: {t.score:0.2f}, score/tok: {t.score/t.length:0.2f}")

			if t.score>best_score:
				best_score = t.score
				best_tracklet = t


	## -- Plot points -- ##
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('x (m)')
	ax.set_ylabel('y (m)')
	ax.set_zlabel('z (m)')
	ax.set_xlim(-11E-1/2, 11E-1/2)
	ax.set_ylim(0, 24E-1)
	ax.set_zlim(0, 3E-1)

	from scipy.optimize import curve_fit

	x_points = []
	y_points = []
	z_points = []

	if best_tracklet is None:
		quit()

	for i, tok in enumerate(best_tracklet.tokens):
		x_points.append(tok.coords[X])
		y_points.append(tok.coords[Y])
		z_points.append(tok.coords[Z])
		# print(tok.f)
		# ax.scatter(xs=tok.coords[X],ys=tok.coords[Y],zs=tok.coords[Z])
		# plt.savefig(f"{i:04d}.png")

	ax.scatter(xs=x_points,ys=y_points,zs=z_points,c=np.arange(len(x_points)), cmap='winter')

	# for t in tracklet_box.tracklets:
	# 	if t is not best_tracklet and t.is_valid:
	# 		for tok in t.tokens:
	# 			ax.scatter(xs=tok.coords[X],ys=tok.coords[Y],zs=tok.coords[Z])

	def func(t,a,b,c,d):
	    return a+b*t+c*t**2+d*t**3

	def d1_func(t,a,b,c,d):
		return b+2*c*t+3*d*t**2

	t = np.linspace(best_tracklet.start_frame*1/90, \
					(best_tracklet.start_frame+best_tracklet.length)*1/90, \
					best_tracklet.length)

	x_params, covmatrix = curve_fit(func, t, x_points)
	y_params, covmatrix = curve_fit(func, t, y_points)
	z_params, covmatrix = curve_fit(func, t, z_points)

	t = np.linspace(0,2,1000)

	x_est = func(t,*x_params)
	y_est = func(t,*y_params)
	z_est = func(t,*z_params)

	xd1_est = d1_func(t,*x_params)
	yd1_est = d1_func(t,*y_params)
	zd1_est = d1_func(t,*z_params)

	bounce_pos = len(z_est[z_est>0])-1
	x_vel = xd1_est[bounce_pos]
	y_vel = yd1_est[bounce_pos]
	z_vel = zd1_est[bounce_pos]

	print(x_vel,y_vel,z_vel)
	print(f"velocity: {np.sqrt(x_vel**2+y_vel**2+z_vel**2):2.2f} m/s")

	z_est[z_est<0] = None


	ax.plot3D(x_est,y_est,z_est)
	plt.show()