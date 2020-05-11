PI = 3.141592653589793
DEBUG = False

#################### - Shared Functions - ####################

# Print function for debug messages
def print_debug(my_print):
    if DEBUG:
        print(my_print)

# Converts km per hour to metre per second
def kph_2_mps(kph):
	return kph*10/36

# Converts metre per second to km per hour
def mps_2_kph(mps):
	return mps*3.6

# Converts degrees to radians
def deg_2_rad(angle_deg):
	return angle_deg*PI/180

# Converts radians to degrees
def rad_2_deg(angle_rad):
	return angle_rad*180/PI


#################### - Socket Consts - ####################
SERVER = 'server'
CLIENT = 'client'
LEFT_CLIENT = 'left'
RIGHT_CLIENT = 'right'

SOCKET_TIMEOUT = 1
HEADER_LENGTH = 10
SERVER_IP = "192.168.20.10"		# server IP address
PORT = 1234 

CHUNK_SIZE = 4096

#################### - Message Consts - ####################
TYPE_STR = "str"
TYPE_VAR = "var"
TYPE_RECORD = "record"
TYPE_CAP = "cap"
TYPE_IMG = "img"
TYPE_BALLS = "balls"
TYPE_DONE = "done"
TYPE_START_CAM = "start_cam"
TYPE_SHUTDOWN = "shutdown"
TYPE_STREAM = "stream"

#################### - Camera Consts - ####################
REC_T = 1					# default recording time
REC_T_MAX = 5				# maximum recording time
CALIB_T = 10  				# number of calibration images to take
CALIB_IMG_DELAY = 2 		# seconds between each image
STREAM_T_MAX = 60 			# maximum time for a stream
STREAM_IMG_DELTA = 15		# number of frames between each streamed image

FRAMERATE = 90
FRAME_HEIGHT = 480
FRAME_WIDTH = 640
RESOLUTION = (FRAME_WIDTH, FRAME_HEIGHT)
FRAME_SIZE = (FRAME_HEIGHT, FRAME_WIDTH)
ISO = 400
CAM_BASELINE = 25E-2
CAM_HEIGHT = 42.5E-2
CAM_ANGLE = deg_2_rad(-6)

################## - Processing Consts - ##################
N_PROCESSORS = 4
LEARNING_RATE = 0.15
BACKGROUND_RATE = 30	# number of frames between each background mean/std dev update
FOREGROUND_SENS = 3		# how sensitive the image differencing is to changes (higher is more sensitive)
N_OBJECTS = 50
BALL_SIZE_MIN = 2
BALL_SIZE_MAX = 1000
WIDTH = 0
HEIGHT = 1
SIZE = 2
X_COORD = 3
Y_COORD = 4
LPF_THRESH = 150
DISP_Y = 30
SIM_THRESH = 0.1

#################### - Graph Consts - ####################
X_3D = 0
Y_3D = 1
Z_3D = 2
VM = 150	# max ball velocity
dT = 1/FRAMERATE	# inter frame time
dM = kph_2_mps(VM)*dT 	# max dist
C_INIT = 0
C_CAND = 1
MAX_EST = 3
EXTRAPOLATE_N = 3
TRACKLET_SCORE_THRESH = 1
TOKEN_SIM_THRESH = dM/2
TOKEN_SCORE_THRESH = 1
SCORE_TOK_THRESH = 1
thetaM = PI
phiM = PI

dT = 1/FRAMERATE	# inter frame time
dM = kph_2_mps(VM)*dT 	# max dist

#################### - LED Consts - ####################
LED_F_MAX = 60				# max LED frequency
LED_F_MIN = 0.5				# min LED frequency
R_LED_F = 0					# default LED frequency
G_LED_F = 0		
R_LED_PIN = 24				# red LED pin
G_LED_PIN = 23				# green LED pin

#################### - Calibration Consts - ####################
SENSOR_SIZE = (3.68, 2.76)	# size of the image sensor on the camera
SQUARE_SIZE = 23.4E-3		# size of squares on the chessboard
# SQUARE_SIZE = 1/3			# size of squares on the chessboard
PATTERN_SIZE = (9, 6)  		# number of points (where the black and white intersects)
MIN_PATTERNS = 15

#################### - Filename Consts - ####################
STEREO_CALIB_F = 'stereo_calib.npy'
LEFT_CALIB_F = 'calib_L.npy'
RIGHT_CALIB_F = 'calib_R.npy'

ACTIVE_CALIB_F = '2020-04-03_calib'
ACTIVE_STEREO_F = '0.4008stereo_calib.npy'	# filename of stereo calib

#################### - Directory Consts - ####################
IMG_DIR = 'img'
DATA_DIR = 'data'
CALIB_DIR = 'calib'
STEREO_CALIB_DIR = 'stereo_calib'
CALIB_IMG_L_DIR = 'calib_L'
CALIB_IMG_R_DIR = 'calib_R'
CALIB_IMG_S_DIR = 'calib_S'

# import os
# ROOT_P = os.getcwd()
# IMG_P = ROOT_P + '//' + IMG_F
# DATA_P = ROOT_P + '//' + DATA_F
# CALIB_P = DATA_P + '//' + CALIB_F
# ACTIVE_CALIB_P = CALIB_P + '//' + ACTIVE_CALIB_F
# STEREO_CALIB_P = DATA_P + '//' + S_CALIB_F

# LEFT_CALIB_IMG_P = IMG_P + '//' + CALIB_IMG_L
# RIGHT_CALIB_IMG_P = IMG_P + '//' + CALIB_IMG_R
# STEREO_CALIB_IMG_P = IMG_P + '//' + CALIB_IMG_S