#################### - funcs.py - ####################
'''
Stores basic functions that are shared across all tennis trainer scripts
'''
##########################################################
import consts as c
import os

def print_debug(my_print):
	'''
	Print function for debug messages
	'''
	if c.DEBUG:
		print(my_print)

def kph_2_mps(kph):
	'''
	Converts km per hour to metre per second
	'''
	return kph*10/36

def mps_2_kph(mps):
	'''
	Converts metre per second to km per hour
	'''
	return mps*3.6

def deg_2_rad(angle_deg):
	'''
	Converts degrees to radians
	'''
	return angle_deg*c.PI/180

def rad_2_deg(angle_rad):
	'''
	Converts radians to degrees
	'''
	return angle_rad*180/c.PI

def make_path(root, *args):
	'''
	Creates a path or args with starting with root separated by //
	'''
	path = root
	for i, arg in enumerate(args):
	    path+=("//" + arg)

	return path

def clean_dir(path):
	root_p = os.getcwd()
	try:
		os.chdir(path)
		for file in os.listdir():
			os.remove(file)
		os.chdir(root_p)
		return True

	except OSError as e:
		print(e)
		return False
