#################### - shared_funcs.py - ####################
'''
Stores basic functions that are shared across all tennis trainer scripts
'''
##########################################################
import consts as c

# Print function for debug messages
def print_debug(my_print):
    if c.DEBUG:
        print(my_print)

# Converts km per hour to metre per second
def kph_2_mps(kph):
	return kph*10/36

# Converts metre per second to km per hour
def mps_2_kph(mps):
	return mps*3.6

# Converts degrees to radians
def deg_2_rad(angle_deg):
	return angle_deg*c.PI/180

# Converts radians to degrees
def rad_2_deg(angle_rad):
	return angle_rad*180/c.PI