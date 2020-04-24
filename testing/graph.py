import numpy as np
import copy

class Tracklet(object):
	def __init__(self, start_frame, tracklet_box=None, tokens=[], score=0,length=0):
		self.start_frame = start_frame
		self.tracklet_box = tracklet_box
		self.tokens = tokens
		self.score = score
		self.length = length
		self.con_est = 0
		self.is_valid = True

tracklets = []

for i in range(10):
	tracklet = Tracklet(i, score=i+1)
	tracklets.append(tracklet)

graph = {	0: [2,3],
			1: [4],
			2: [4],
			3: [],
			4: [6,7,8],
			5: [],
			6: [],
			7: [9],
			8: [],
			9: []}

# -- get all graph nodes that have no connections
start_nodes = [x for x in range(len(tracklets))]
end_nodes = []

for node, conn in graph.items():
	if conn == []:
		end_nodes.append(node)
	for c in conn:
		if c in start_nodes:
			start_nodes.remove(c)

longest_path = {}

def get_longest_path(graph, start, end=None, path=[], score=0):

	if start not in path:
		score-= tracklets[start].score
		path.append(start)
		if score < longest_path[start]['score']:
				longest_path[start]['path'] = copy.copy(path)
				longest_path[start]['score'] = -score

	for node in graph[start]:
		if node not in path:
			get_longest_path(graph, node, end, path, score)

	del path[-1]

for node_s in start_nodes:
	for node, conn in graph.items():
		longest_path[node] = {'score':0, 'path':[]}

	get_longest_path(graph, node_s)
	
	for node_e in end_nodes:
		print(longest_path[node_e])