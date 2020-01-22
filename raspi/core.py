import cv2
import picamera
import time
import camera as my_cam
import multiprocessing as mp
from queue import Empty

# constants
INITIALISING = 0
RECORDING = 1
READY = 2

# control variables
debug = True
state = INITIALISING

# used for printing debug messages
def print_dbg(message):
	if debug:
		print(message)
	return

if __name__ == "__main__":
	#### ---- initialise objects ---- ####
	queue = mp.Queue()
    finished = mp.Event()
	output = QueueOutput(queue, finished)

	#### ---- initialise processors ---- ####
	capture_proc = mp.Process(target=capture, args=(queue, finished))
    processing_procs = [
        mp.Process(target=do_processing, args=(queue, finished))
        for i in range(4)
        ]
    for proc in processing_procs:
        proc.start()
    capture_proc.start()

	#### ---- initialise camera ---- ####
	print_dbg('initialising camera...')
	with picamera.PiCamera(resolution='VGA', framerate=90) as camera:
		camera.vflip=True
		camera.annotate_frame_num = True
		camera.annotate_text_size = 160
		camera.annotate_foreground = picamera.Color('green')
		time.sleep(2)
		camera.start_recording(output, format='mjpeg')
		print_dbg('camera initialised')