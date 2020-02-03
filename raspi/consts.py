DEBUG = False

SERVER = 'server'
CLIENT = 'client'

LEFT_CLIENT = 'left'
RIGHT_CLIENT = 'right'

SOCKET_TIMEOUT = 5
HEADER_LENGTH = 24
IP = "192.168.20.8"		# server IP address
# IP = "127.0.0.1"
PORT = 1234

CHUNK_SIZE = 4096 

## -- message type definitions -- ##
TYPE_STR = "text"
TYPE_VAR = "var"

TYPE_REC = "record"
TYPE_CAP = "capture"
TYPE_IMG = "img"
TYPE_BALLS = "balls"
TYPE_DONE = "done"

## -- State definitions -- ##
STATE_IDLE = "idle"
STATE_RECORDING = "recording"
STATE_CAPTURING = "capturing"
STATE_STOP = "stop"
STATE_SHUTDOWN = "shutdown"