## -- Shared Functions -- ##
def print_debug(my_print):
    if DEBUG:
        print(my_print)

def kph_2_mps(kph):
	return kph*10/36

def mps_2_kph(mps):
	return mps*3.6

def deg_2_rad(angle_deg):
	return angle_deg*np.pi/180