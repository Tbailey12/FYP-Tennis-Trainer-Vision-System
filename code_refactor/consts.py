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

def find_tennis_court():
	x_coords, y_coords = [], []
	# draw centremark
	x_coords.extend([0,	0, 0])
	y_coords.extend([0, TENNIS_CENTRE_MARK, 0])

	# draw outside
	x_coords.extend([TENNIS_WIDTH/2, TENNIS_WIDTH/2, -TENNIS_WIDTH/2, -TENNIS_WIDTH/2, 0])
	y_coords.extend([0, TENNIS_LENGTH, TENNIS_LENGTH, 0, 0])

	# draw sideline (top)
	top_sideline = TENNIS_WIDTH/2-TENNIS_SIDELINE
	x_coords.extend([top_sideline, top_sideline, 0])
	y_coords.extend([0, TENNIS_LENGTH, TENNIS_LENGTH])

	# draw centremark
	x_coords.extend([0, 0])
	y_coords.extend([TENNIS_LENGTH-TENNIS_CENTRE_MARK, TENNIS_LENGTH])

	# draw sideline (bottom)
	bottom_sideline = -TENNIS_WIDTH/2+TENNIS_SIDELINE
	x_coords.extend([bottom_sideline, bottom_sideline, top_sideline, bottom_sideline])
	y_coords.extend([TENNIS_LENGTH, TENNIS_LENGTH-TENNIS_SERVICE, TENNIS_LENGTH-TENNIS_SERVICE, TENNIS_LENGTH-TENNIS_SERVICE])

	# draw net
	x_coords.extend([bottom_sideline, -TENNIS_WIDTH/2-TENNIS_NET_POST, TENNIS_WIDTH/2+TENNIS_NET_POST, bottom_sideline])
	y_coords.extend([TENNIS_LENGTH/2, TENNIS_LENGTH/2, TENNIS_LENGTH/2, TENNIS_LENGTH/2])

	# draw right and centre service
	x_coords.extend([bottom_sideline, bottom_sideline, top_sideline, 0, 0])
	y_coords.extend([0,TENNIS_SERVICE, TENNIS_SERVICE, TENNIS_SERVICE, TENNIS_LENGTH-TENNIS_SERVICE])

	z_coords = [0 for x in x_coords]
	return x_coords, y_coords, z_coords

#################### - Socket Consts - ####################
SERVER = 'server'
CLIENT = 'client'
LEFT_CLIENT = 'left'
RIGHT_CLIENT = 'right'

SOCKET_TIMEOUT = 1
HEADER_LENGTH = 10
SERVER_IP = "192.168.1.20"		# server IP address
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
TYPE_HELP = "help"

#################### - Camera Consts - ####################
FRAMERATE = 90
FRAME_HEIGHT = 480
FRAME_WIDTH = 640
STREAM_SCALER = 0.5
RESOLUTION = (FRAME_WIDTH, FRAME_HEIGHT)
FRAME_SIZE = (FRAME_HEIGHT, FRAME_WIDTH)
ISO = 200
SHUTTER_SPEED = 2000		# shutter speed in microseconds
CAM_BASELINE = 40.2E-2
CAM_HEIGHT = 35E-2
CAM_ANGLE = deg_2_rad(-10.5)

REC_T = 1					# default recording time
REC_T_MAX = 3				# maximum recording time
STREAM_T_MAX = 60 			# maximum time for a stream
STREAM_IMG_DELTA = 15		# number of frames between each streamed image
STREAM_DELTA_T = int(100*STREAM_IMG_DELTA/FRAMERATE)

################## - Processing Consts - ##################
N_PROCESSORS = 4
LEARNING_RATE = 0.05
BACKGROUND_RATE = 5		# number of frames between each background mean/std dev update
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
DISP_Y = 50
SIM_THRESH = 0.1
SIM_WR_THRESH = 2
SIM_HR_THRESH = 2
SIM_SR_THRESH = 2

#################### - Graph Consts - ####################
SCALER = 1E-1
X_3D = 0
Y_3D = 1
Z_3D = 2
VM = 150*SCALER			# max ball velocity
dT = 1/FRAMERATE		# inter frame time
dM = kph_2_mps(VM)*dT*2 # max dist
WIN_SIZE = 30
WIN_OVERLAP = 5
MAX_EST = 3
CAND_INIT = 0
CAND_DATA = 1
EXTRAPOLATE_N = 5
MAX_SHARED_TOKS = 5
MIN_SHARED_TOKS = 3
EPSILON = 1E-6
thetaM = PI
phiM = PI
TRACKLET_SCORE_THRESH = 1
TOKEN_SIM_THRESH = dM
TOKEN_SCORE_THRESH = 1
SCORE_TOK_THRESH = 1

X_CORRECTION = 0E-2
Y_CORRECTION = 0E-2
Z_CORRECTION = 0E-2

ZMIN = 0
ZMAX = 3*SCALER
YMIN = 4*SCALER
YMAX = 24*SCALER
XMIN = -11*SCALER
XMAX = 11*SCALER

FIT_POINTS = 1000
SHOT_T_MAX = 3

TENNIS_SIDELINE = 1.37*SCALER
TENNIS_WIDTH = 10.97*SCALER
TENNIS_LENGTH = 23.77*SCALER
TENNIS_SERVICE = 5.485*SCALER
TENNIS_CENTRE_MARK = 0.3*SCALER
TENNIS_NET_POST = 0.91*SCALER

TENNIS_X_POINTS, TENNIS_Y_POINTS, TENNIS_Z_POINTS = find_tennis_court()

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
PATTERN_SIZE = (9, 6)  		# number of points (where the black and white intersects)
MIN_PATTERNS = 15

#################### - Filename Consts - ####################
STEREO_CALIB_F = 'stereo_calib.npy'
LEFT_CALIB_F = 'calib_L.npy'
RIGHT_CALIB_F = 'calib_R.npy'

ACTIVE_CALIB_DIR = '2020-04-03_calib'
ACTIVE_STEREO_F = '0.4479stereo_calib.npy'	# filename of stereo calib

#################### - Directory Consts - ####################
IMG_DIR = 'img'
DATA_DIR = 'data'
CALIB_DIR = 'calib'
RECORD_DIR = 'record'
STREAM_DIR = 'stream'
STEREO_CALIB_DIR = 'stereo_calib'
CALIB_IMG_L_DIR = 'calib_L'
CALIB_IMG_R_DIR = 'calib_R'
CALIB_IMG_S_DIR = 'calib_S'