import numpy as np
import os
import time
import copy
import graph_utils as gr

ROOT_P = 'D:\\documents\\local uni\\FYP\\code'

SCALER = 1

X = 0
Y = 1
Z = 2
FPS = 90
VM = 150*SCALER	# max ball velocity
# Vm = 30*SCALER		# min ball velocity


WIN_SIZE = 30
WIN_OVERLAP = 5

ZMIN = 0
ZMAX = 3.5*SCALER
YMIN = 4*SCALER
YMAX = 24*SCALER
XMIN = -11*SCALER
XMAX = 11*SCALER

MAX_EST = 3
dT = 1/FPS	# inter frame time
CAND_INIT = 0
CAND_DATA = 1
EXTRAPOLATE_N = 5
MAX_SHARED_TOKS = 5
MIN_SHARED_TOKS = 3

EPSILON = 1E-6

def kph_2_mps(kph):
	return kph*10/36

dM = kph_2_mps(VM)*dT*2 	# max dist
thetaM = np.pi
phiM = np.pi

TRACKLET_SCORE_THRESH = 1
TOKEN_SIM_THRESH = dM
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

	def merge_tracklets(self):
		graph = gr.create_graph(self.tracklets)

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
		for t in range(len(self.tracklets)-1):
			cons_count_max = 0
			cons_pos_max = None
			for tok1 in reversed(self.tracklets[t].tokens[-MAX_SHARED_TOKS:]):
				cons_count = 0
				cons_pos = 0
				cons = False
				for tok2 in self.tracklets[t+1].tokens[:MAX_SHARED_TOKS]:
					if tok1.f == tok2.f:
						sim = tok1.calc_similarity(tok2)
					else:
						continue

					if sim < TOKEN_SIM_THRESH:
						cons = True
						cons_count += 1
						cons_pos = self.tracklets[t+1].tokens.index(tok2)
					else:
						break

					if cons == True and cons_count > cons_count_max:
						cons_pos_max = cons_pos

			if cons_pos_max is not None:
				graph[t].append(t+1)
				for i, tok in enumerate(self.tracklets[t+1].tokens):
					if i<=cons_pos_max:
						self.tracklets[t+1].score -= self.tracklets[t+1].tokens[i].score
						self.tracklets[t+1].tokens[i].score = 0

			if cons_pos_max is None:
				if self.tracklets[t].length > 3 and self.tracklets[t+1].length > 3:
					first_extrapolation_points = []
					second_extrapolation_points = []

					for i in range(3):
						first_extrapolation_points.append(self.tracklets[t].tokens[i-3].coords)
						second_extrapolation_points.append(self.tracklets[t+1].tokens[2-i].coords)

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
							self.tracklets[t].add_token(Token(self.tracklets[t].tokens[-1].f+1,first_point,score=1))

						for second_point in reversed(new_second_points):
							self.tracklets[t].add_token(Token(self.tracklets[t].tokens[-1].f+1,second_point,score=1))

						graph[t].append(t+1)

		start_nodes, end_nodes = gr.get_start_end_nodes(graph)

		for item in graph.items():
			print(item)

		longest_path = {}
		path_list = []
		for node_s in start_nodes:
			for node, conn in graph.items():
				longest_path[node] = {'score':0, 'path':[]}

			gr.get_longest_paths(self.tracklets, longest_path, graph, node_s)
			
			for node_e in end_nodes:
				path_list.append(longest_path[node_e])

		score = 0
		best_path = None
		for path in path_list:
			if path['score'] > score:
				score=path['score']
				best_path = path

		if best_path is not None:
			merged_track = Tracklet(start_frame=self.tracklets[best_path['path'][0]].start_frame)

			f=-1
			print(best_path)
			for t in best_path['path']:
				for tok in self.tracklets[t].tokens:
					if tok.f > f:
						merged_track.add_token(tok)
						f = tok.f
				self.tracklets[t].is_valid = False

			self.add_tracklet(merged_track)

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
		if self.tracklet_box is not None:
			self.tracklet_box.add_tracklet(Tracklet(start_frame = copy.deepcopy(self.start_frame),\
						tokens = copy.deepcopy(self.tokens), score = copy.deepcopy(self.score), \
						length = copy.deepcopy(self.length)))

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
	print(f)
	if f < f_max:
		est = tracklet.est_next()

		# are there tokens to compare?
		# yes: compare the tokens and continue
		# no: add the estimate as a token and continue
		valid_cand = False
		if candidates_3D[f] != []:
			valid_cand = False
			for i, cand in enumerate(candidates_3D[f]):
				c4 = cand[CAND_DATA]
				candidates_3D[f][i][CAND_INIT] = True
				score = score_node(est, c4)
				if score > TOKEN_SCORE_THRESH:
					valid_cand = True
					if f_max-WIN_SIZE <= f:
						tracklet.add_token(Token(f, c4, score))
					else:
						tracklet.add_token(Token(f, c4, 0))

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
	for c in [c1,c2,c3]:
		if 	c[X] > XMAX or c[X] < XMIN \
		or	c[Y] > YMAX or c[Y] < YMIN \
		or	c[Z] > ZMAX or c[Z] < ZMIN:
			return False

	d1 = calc_dist(c2-c1)
	d2 = calc_dist(c3-c2)

	if d1<dM and d2<dM and d1>0 and d2>0:
		return True
	else:
		return False


def get_tracklets(candidates_3D):
	for f, frame in enumerate(candidates_3D):
		for c, candidate in enumerate(frame):
			candidates_3D[f][c] = [False, np.array(candidate)]

	## -- Shift Token Transfer -- ##
	num_frames = len(candidates_3D)
	tracklet_box = TrackletBox()

	for f in range(num_frames):
		win_start = 0
		win_end = 0

		if f == 0:
			win_start = 0
			win_end = win_start+WIN_SIZE

		elif f % WIN_SIZE == 0 and f != 0:
			win_start = f-WIN_OVERLAP
			win_end = f+WIN_SIZE

			if win_end > num_frames:
				win_end = num_frames

			for frame in candidates_3D:
				for cand in frame:
					cand[CAND_INIT] = False
		
		else:
			continue

		init_set = False
		c1,c2,c3 = [],[],[]

		for cur_frame in range(win_start+3, win_end):
			if init_set is False:
				c1 = candidates_3D[cur_frame-3]
				c2 = candidates_3D[cur_frame-2]
				c3 = candidates_3D[cur_frame-1]

				if (c1 == []) or (c2 == []) or (c3 == []):
					continue
				else:
					init_set = True

			if init_set:
				tracklet = Tracklet(cur_frame-3, tracklet_box)
				for c1_c in c1:
					if c1_c[CAND_INIT] is True:	continue
					tracklet.add_token(Token(cur_frame-3,c1_c[CAND_DATA], score=0))
					for c2_c in c2:
						if c2_c[CAND_INIT] is True:	continue
						tracklet.add_token(Token(cur_frame-2,c2_c[CAND_DATA], score=0))
						for c3_c in c3:
							if c3_c[CAND_INIT] is True:	continue
							tracklet.add_token(Token(cur_frame-1,c3_c[CAND_DATA], score=0))

							c1_c[CAND_INIT] = True
							c2_c[CAND_INIT] = True
							c3_c[CAND_INIT] = True

							if check_init_toks(c1_c[CAND_DATA],c2_c[CAND_DATA],c3_c[CAND_DATA]):
								evaluate(candidates_3D, tracklet, cur_frame, f_max=win_end)				

							tracklet.del_token()
						tracklet.del_token()
					tracklet.del_token()

				init_set = False
				c1,c2,c3 = [],[],[]

	tracklet_box.merge_tracklets()

	return tracklet_box

def find_best_tracklet(tracklet_box):
	best_score, best_tracklet = 0, None
	for t in tracklet_box.tracklets:
		if t.is_valid:
			# print(f"f_start: {t.start_frame}, f_end: {t.start_frame+t.length}, score: {t.score:0.2f}, score/tok: {t.score/t.length:0.2f}")

			if t.score>best_score:
				best_score = t.score
				best_tracklet = t

	return best_tracklet

def curve_func(t,a,b,c,d):
	return a+b*t+c*t**2+d*t**3

def d1_curve_func(t,a,b,c,d):
	return b+2*c*t+3*d*t**2

def split_tracklet(tracklet):
		acc = []
		vel = []
		for i, tok in enumerate(tracklet.tokens):
			if i==0:
				vel.append(0*tok.coords)
			else:
				vel.append(tracklet.tokens[i].coords-tracklet.tokens[i-1].coords)

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
					new_track = Tracklet(start_frame=0, tracklet_box=None, tokens=[], score=0,length=0)
					for tok in tracklet.tokens[split_start_f:k]:
						new_track.add_token(tok)
				
					return new_track

if __name__ == "__main__":
	RESOLUTION = (640,480)
	w,h = RESOLUTION

	os.chdir(ROOT_P + '\\' + 'img\\simulation_tests')

	candidates_3D = np.load('candidates_3D.npy', allow_pickle=True)

	# -- Plot points -- ##
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure(figsize=(15*1.25,4*1.25))
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('x (m)')
	ax.set_ylabel('y (m)')
	ax.set_zlabel('z (m)')
	ax.set_xlim(XMIN/2, XMAX/2)
	ax.set_ylim(0, YMAX)
	ax.set_zlim(0, 2)
	ax.view_init(elev=20,azim=-20)

	# for frame in candidates_3D:
	# 	for cand in frame:
	# 		ax.scatter(xs=cand[0],ys=cand[1],zs=cand[2])
	
	# plt.show()
	# quit()

	tracklet_box = get_tracklets(candidates_3D)
	best_tracklet = find_best_tracklet(tracklet_box)

	if best_tracklet is None:
		print('no valid trajectories found')
		quit()

	best_tracklet = split_tracklet(best_tracklet)

	# print(best_tracklet.length)
	# print(len(best_tracklet.tokens))

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

	for tracklet in tracklet_box.tracklets:
			for tok in tracklet.tokens:
				ax.scatter(xs=tok.coords[X],ys=tok.coords[Y],zs=tok.coords[Z],c='blue',alpha=0.2)

	# ax.scatter(xs=x_points,ys=y_points,zs=z_points,c=np.arange(len(x_points)), cmap='winter')

	t = np.linspace(best_tracklet.start_frame*1/90, \
					(best_tracklet.start_frame+best_tracklet.length)*1/90, \
					best_tracklet.length)

	x_params, covmatrix = curve_fit(curve_func, t, x_points)
	y_params, covmatrix = curve_fit(curve_func, t, y_points)
	z_params, covmatrix = curve_fit(curve_func, t, z_points)

	t = np.linspace(0,2,1000)

	X_OFFSET = 0.016436118692901215
	Y_OFFSET = 0.6083691217642057
	Z_OFFSET = -0.04876521114374302

	x_params[0]-=X_OFFSET
	y_params[0]-=Y_OFFSET
	z_params[0]-=Z_OFFSET

	x_est = curve_func(t,*x_params)
	y_est = curve_func(t,*y_params)
	z_est = curve_func(t,*z_params)

	xd1_est = d1_curve_func(t,*x_params)
	yd1_est = d1_curve_func(t,*y_params)
	zd1_est = d1_curve_func(t,*z_params)

	bounce_pos = 0
	for i, z in enumerate(z_est):
		bounce_pos = i
		if z<=0:
			bounce_pos = i
			break
	# bounce_pos = len(z_est[z_est>0])-1

	x_vel = xd1_est[bounce_pos]
	y_vel = yd1_est[bounce_pos]
	z_vel = zd1_est[bounce_pos]

	print(x_vel,y_vel,z_vel)
	print(f"velocity: {np.sqrt(x_vel**2+y_vel**2+z_vel**2):2.2f} m/s")
	print(f"bounce_loc: {x_est[bounce_pos]:0.2f}, {y_est[bounce_pos]:0.2f},{z_est[bounce_pos]:0.2f}")

	z_est[bounce_pos:] = None

	print(bounce_pos)

	ax.plot3D(x_est,y_est,z_est,c='red')
	plt.show()