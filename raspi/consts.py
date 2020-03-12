DEBUG = False

SERVER = 'server'
CLIENT = 'client'

LEFT_CLIENT = 'left'
RIGHT_CLIENT = 'right'

## -- Socket consts -- ##
SOCKET_TIMEOUT = 5
HEADER_LENGTH = 24
IP = "192.168.20.11"		# server IP address
PORT = 1234 

CHUNK_SIZE = 4096
REC_T = 3					# default recording time
CALIB_T = 10  				# number of calibration images to take
CALIB_IMG_DELAY = 1 		# seconds between each image
STREAM_MAX = 60 			# maximum time for a stream

## -- Camera consts -- ##
FRAMERATE = 90
RESOLUTION = (640,480)

## -- Processing consts -- ##
N_PROCESSORS = 4
LEARNING_RATE = 0.15
BACKGROUND_RATE = 30

## -- LED consts -- ##
LED_F_MAX = 60				# max LED frequency
LED_F_MIN = 0.5				# min LED frequency
R_LED_F = 0					# default LED frequency
G_LED_F = 0		
R_LED_PIN = 24				# red LED pin
G_LED_PIN = 23				# green LED pin

## -- message type definitions -- ##
TYPE_STR = "text"
TYPE_VAR = "var"

TYPE_REC = "record"
TYPE_CAP = "capture"
TYPE_IMG = "img"
TYPE_BALLS = "balls"
TYPE_DONE = "done"
TYPE_SHUTDOWN = "shutdown"
TYPE_STREAM = "stream"

## -- Calibration consts -- ##
SENSOR_SIZE = (3.68, 2.76)	# size of the image sensor on the camera
SQUARE_SIZE = 23.4E-3		# size of squares on the chessboard
PATTERN_SIZE = (9, 6)  		# number of points (where the black and white intersects)
MIN_PATTERNS = 25

## -- Filename consts -- ##
STEREO_CALIB_F = 'stereo_calib.npy'
LEFT_CALIB_F = 'calib_L.npy'
RIGHT_CALIB_F = 'calib_R.npy'

## -- Directory consts -- ##
IMG_F = 'img'
DATA_F = 'data'
CALIB_F = 'calib'
S_CALIB_F = 'stereo_calib'
CALIB_IMG_L = 'calib_L'
CALIB_IMG_R = 'calib_R'
CALIB_IMG_S = 'calib_S'

## -- Active calibration consts -- ##
ACTIVE_CALIB_F = '2020-03-12_calib2'		# directory containing L,R calib
ACTIVE_STEREO_F = '0.4545stereo_calib.npy'	# filename of stereo calib

import os
ROOT_P = os.getcwd()
IMG_P = ROOT_P + '//' + IMG_F
DATA_P = ROOT_P + '//' + DATA_F
CALIB_P = DATA_P + '//' + CALIB_F
ACTIVE_CALIB_P = CALIB_P + '//' + ACTIVE_CALIB_F
STEREO_CALIB_P = DATA_P + '//' + S_CALIB_F

LEFT_CALIB_IMG_P = IMG_P + '//' + CALIB_IMG_L
RIGHT_CALIB_IMG_P = IMG_P + '//' + CALIB_IMG_R
STEREO_CALIB_IMG_P = IMG_P + '//' + CALIB_IMG_S

## -- Shared Functions -- ##
def print_debug(my_print):
    if DEBUG:
        print(my_print)