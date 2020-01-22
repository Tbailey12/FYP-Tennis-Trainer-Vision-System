import io
import time
import threading
import picamera
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import multiprocessing as mp
import queue
import cv2
import os
import ctypes

# w,h = (1280,720)
w,h = (640,480)
resolution = w,h
framerate = 90
t_record = 1    # seconds to record
n_processors = 4    # number of processors to use for CV

def ImageProcessor(unprocessed_frames, processed_frames, recording, proc_complete):
    processing = False

    while True:
        if not processing:
            recording.wait()    # wait for a recording to start
            processing = True
        else:
            try:
                # get the frames from queue
                n_frame, frame_buf = unprocessed_frames.get_nowait()
                # y_data is a numpy array hxw with 8 bit greyscale brightness values
                y_data = np.frombuffer(frame_buf, dtype=np.uint8, count=w*h).reshape((h,w))
                # do some processing

            except queue.Empty:
                if not recording.is_set():  # if the recording has finished
                    processing = False      # reset the processing flag
                    proc_complete.set()     # set the proc_complete event

class FrameController(object):
    def __init__(self, unprocessed_frames, processed_frames, recording, proc_complete):
        self.unprocessed_frames = unprocessed_frames
        self.processed_frames = processed_frames
        self.recording = recording
        self.proc_complete = proc_complete
        self.number_of_processors = n_processors
        self.processors = []
        self.n_frame = 0

        ## -- create processors for processing the individual video frames
        for i in range(n_processors):
            proc_event = mp.Event()
            processor = mp.Process(target=ImageProcessor, args=(self.unprocessed_frames,self.processed_frames,self.recording,proc_event))
            self.proc_complete.append(proc_event)
            self.processors.append(processor)
            processor.start()

    def write(self, buf):
        if recording.is_set():
            self.unprocessed_frames.put((self.n_frame, buf))    # add the new frame to the queue
            self.n_frame += 1   # increment the frame number
        else:
            # print(self.n_frame)
            self.n_frame = 0    # reset frame number when recording is finished

    def flush(self):
        for i, processor in enumerate(self.processors):
            self.proc_complete[i].wait() # wait until process has finished
            processor.terminate()
            processor.join()

def StartPicam(unprocessed_frames, processed_frames, recording, shutdown, picam_ready, processing_complete):
    proc_complete = []     # contains processing events, true if complete
    with picamera.PiCamera() as camera:
        camera.framerate = framerate
        camera.resolution = resolution
        camera.vflip = True # camera is upside down so flip the image
        ## -- adds the frame number to the image for testing
        camera.annotate_frame_num = True
        camera.annotate_text_size = 160
        camera.annotate_foreground = picamera.Color('green')
        ## ----------------------------------------------- ##
        time.sleep(2)   # give the camera a couple of seconds to initialise

        output = FrameController(unprocessed_frames, processed_frames, recording, proc_complete)
        camera.start_recording(output, format='yuv')

        picam_ready.set()   # the picam is ready to record

        # Picam will stay in this while loop waiting for either shutdown or recording flags to be set
        while True:
            if recording.wait(1):    # wait for recording flag to be set in main()
                try:
                    camera.wait_recording(t_record) # record for an amount of time
                finally:
                    recording.clear()   # clear the record flag to stop processing
                    for proc in proc_complete:  # wait for all processing to be complete
                        proc.wait()
                    processing_complete.set()   # set processing complete event
                    for proc in proc_complete:  # reset all processing complete events
                        proc.clear()
            elif shutdown.is_set():
                break
        camera.stop_recording()
    return



if __name__ == "__main__":
    ## -- setup client connection to server -- ##

    ## -- initialise multithreading objs -- ##
    unprocessed_frames = mp.Queue()
    processed_frames = mp.Queue()
    recording = mp.Event()
    shutdown = mp.Event()
    picam_ready = mp.Event()
    processing_complete = mp.Event()

    ## -- initialise Picam process for recording
    Picam = mp.Process(target=StartPicam, args=(unprocessed_frames, processed_frames, recording, shutdown, picam_ready, processing_complete))
    Picam.start()

    while True:
        ## -- read server messages -- ##

        # listen for start recording command
        ## for testing
        #### ---- WHEN THE SERVER SAYS TO RECORD ---- ####
        picam_ready.wait()  # waits for the picam to initialise
        # for proc_status in proc_complete:
        #     proc_status.clear()
        recording.set()
        processing_complete.clear()
        print('recording')
        print(time.time())
        #### ---------------------------------------- ####
        print(processing_complete.is_set())
        ## -- send server messages -- ##

        ## for testing

        #### ---- WHEN THE SERVER SAYS TO SHUTDOWN ---- ####
        # for proc_status in proc_complete:
        #     proc_status.wait()  # waits for all processes to be complete

        time.sleep(5)
        # print(proc_complete)
        print(recording.is_set())
        print(processing_complete.is_set())
        # shutdown.set()
        # while Picam.is_alive():
        #     time.sleep(0.1)
        # Picam.join()
        #### ------------------------------------------ ####

        # loop through processed frames and send data to server


#####################################################################################

# def print_l(print_lock, message):
#     print_lock.acquire()
#     try: 
#         print(message)
#     finally:
#         print_lock.release()

# def ImageProcessor(unprocessed_frames, print_lock, process_run):
#     # stream = io.BytesIO()
#     # self.event = multiprocessing.Event()

#     while process_run:
#         try:
#             # attempts to get a new frame, raises exception if none available
#             # continue
#             # print(unprocessed_frames.qsize())

#             frame_num, frame = unprocessed_frames.get_nowait()
#             # stream.write(frame)
#             # stream.seek(0)
#             # Read the image and do some processing on it

#             # im = Image.open(stream) # read image from stream
#             # img = np.array(im) # convert image to numpy array

#             # print_l(print_lock,img)
#             # print_lock.acquire()
#             # cv2.imwrite(f'{frame_num:03d}.png',frame)
#             # cv2.imwrite(f'{frame_num:03.3f}.png',img)
#             # print_lock.release()
#             # Reset the stream
#             # stream.seek(0)
#             # stream.truncate()

#         except queue.Empty:
#             if process_run.value is False:
#                 break
#             continue
#     return
# class ProcessOutput(object):
#     def __init__(self):
#         self.number_of_processors = 4
#         self.unprocessed_frames = multiprocessing.Queue()
#         self.processes = []
#         self.print_lock = multiprocessing.Lock()
#         self.frame_num = 0
#         self.start = time.time()
#         self.done = False
#         self.process_run = multiprocessing.Value(ctypes.c_bool,True)

#         for i in range(self.number_of_processors):
#             processor = multiprocessing.Process(target=ImageProcessor, args=(self.unprocessed_frames, self.print_lock, self.process_run))
#             self.processes.append(processor)
#             processor.start()

#     def write(self, buf):
#         if self.done is False:
#             # y_data = np.frombuffer(buf, dtype=np.uint8, count=w*h).reshape((h,w))
#             # self.unprocessed_frames.put((self.frame_num,y_data)) # add the new frame to the buffer
#             self.frame_num += 1    

#     def flush(self):
#         print('closing camera')
#         # print('flushing')
#         # When told to flush (this indicates end of recording), shut
#         # down in an orderly fashion.
#         print(self.frame_num)
#         for p in self.processes:
#             p.terminate()
#             p.join()

# with picamera.PiCamera() as camera:
#     camera.framerate = framerate
#     camera.resolution = resolution
#     camera.vflip=True
#     camera.annotate_frame_num = True
#     camera.annotate_text_size = 160
#     camera.annotate_foreground = picamera.Color('green')
#     time.sleep(2)

#     output = ProcessOutput()
#     camera.start_recording(output, format='yuv')

#     try:   
#         camera.wait_recording(1)
#     finally:
#         output.process_run.value = False    # ensures all processes stop on next run
#         output.done = True  # stop writing frames to queue

#     while(True):
#         dead_count = 0
#         for process in output.processes:
#             if process.is_alive() is False:
#                 dead_count+=1
#         if dead_count == output.number_of_processors:
#             break

#     camera.stop_recording()

##### ========================================================================== #####

# w,h = (1280,720)
# w,h = (640,480)
# resolution = w,h
# framerate = 90

# def print_l(print_lock, message):
#     print_lock.acquire()
#     try: 
#         print(message)
#     finally:
#         print_lock.release()

# def ImageProcessor(unprocessed_frames, print_lock, process_run):
#     # stream = io.BytesIO()
#     # self.event = multiprocessing.Event()

#     while process_run:
#         try:
#             # attempts to get a new frame, raises exception if none available
#             # print(unprocessed_frames.qsize())

#             frame_num, frame_buf = unprocessed_frames.get_nowait()
#             # y_data = np.frombuffer(frame_buf, dtype=np.uint8, count=w*h).reshape((h,w))
#             # print(y_data)
#             # stream.write(frame)
#             # stream.seek(0)
#             # Read the image and do some processing on it

#             # im = Image.open(stream) # read image from stream
#             # img = np.array(im) # convert image to numpy array

#             # print_l(print_lock,img)
#             # print_lock.acquire()
#             time.sleep(0.02)

#             # cv2.imwrite(f'{frame_num:03d}.png',y_data)
#             # cv2.imwrite(f'{frame_num:03.3f}.png',img)
#             # print_lock.release()
#             # Reset the stream
#             # stream.seek(0)
#             # stream.truncate()

#         except queue.Empty:
#             if process_run.value is False:
#                 break
#             continue
#     return
# class ProcessOutput(object):
#     def __init__(self):
#         self.number_of_processors = 4
#         self.unprocessed_frames = multiprocessing.Queue()
#         self.processes = []
#         self.print_lock = multiprocessing.Lock()
#         self.frame_num = 0
#         self.start = time.time()
#         self.done = False
#         self.process_run = multiprocessing.Value(ctypes.c_bool,True)

#         for i in range(self.number_of_processors):
#             processor = multiprocessing.Process(target=ImageProcessor, args=(self.unprocessed_frames, self.print_lock, self.process_run))
#             self.processes.append(processor)
#             processor.start()

#     def write(self, buf):
#         if self.done is False:
#             self.unprocessed_frames.put((self.frame_num, buf)) # add the new frame to the buffer
#             self.frame_num += 1    

#     def flush(self):
#         print('closing camera')
#         # print('flushing')
#         # When told to flush (this indicates end of recording), shut
#         # down in an orderly fashion.
#         print(self.frame_num/10)
#         for p in self.processes:
#             p.terminate()
#             p.join()

# with picamera.PiCamera() as camera:
#     camera.framerate = framerate
#     camera.resolution = resolution
#     camera.vflip=True
#     camera.annotate_frame_num = True
#     camera.annotate_text_size = 160
#     camera.annotate_foreground = picamera.Color('green')
#     time.sleep(2)

#     output = ProcessOutput()
#     camera.start_recording(output, format='yuv')

#     try:   
#         camera.wait_recording(10)
#     finally:
#         output.process_run.value = False    # ensures all processes stop on next run
#         output.done = True  # stop writing frames to queue

#     while(True):
#         dead_count = 0
#         for process in output.processes:
#             if process.is_alive() is False:
#                 dead_count+=1
#         if dead_count == output.number_of_processors:
#             break

#     camera.stop_recording()

## =================================================================================== ##

    # output = ProcessOutput()
    # camera.start_recording(output, format='mjpeg')
    # print(time.time()-start)

    # try:   
    #     camera.wait_recording(2)
    # finally:
    #     output.process_run.value = False    # ensures all processes stop on next run
    #     output.done = True  # stop writing frames to queue

    # while(True):
    #     dead_count = 0
    #     for process in output.processes:
    #         if process.is_alive() is False:
    #             dead_count+=1
    #     if dead_count == output.number_of_processors:
    #         break

    # print('closing')
    # camera.close()
# import io
# import time
# import threading
# import picamera

# class ImageProcessor(threading.Thread):
#     def __init__(self, owner):
#         super(ImageProcessor, self).__init__()
#         self.stream = io.BytesIO()
#         self.event = threading.Event()
#         self.terminated = False
#         self.owner = owner
#         self.start()

#     def run(self):
#         # This method runs in a separate thread
#         while not self.terminated:
#             # Wait for an image to be written to the stream
#             if self.event.wait(1):
#                 try:
#                     self.stream.seek(0)
#                     # Read the image and do some processing on it
#                     #Image.open(self.stream)
#                     #...
#                     #...
#                     # Set done to True if you want the script to terminate
#                     # at some point
#                     # self.owner.done=True
#                 finally:
#                     # Reset the stream and event
#                     self.stream.seek(0)
#                     self.stream.truncate()
#                     self.event.clear()
#                     # Return ourselves to the available pool
#                     with self.owner.lock:
#                         self.owner.pool.append(self)

# class ProcessOutput(object):
#     def __init__(self):
#         self.done = False
#         # Construct a pool of 4 image processors along with a lock
#         # to control access between threads
#         self.lock = threading.Lock()
#         self.pool = [ImageProcessor(self) for i in range(4)]
#         self.processor = None

#     def write(self, buf):
#         if buf.startswith(b'\xff\xd8'):
#             # New frame; set the current processor going and grab
#             # a spare one
#             if self.processor:
#                 self.processor.event.set()
#             with self.lock:
#                 if self.pool:
#                     self.processor = self.pool.pop()
#                 else:
#                     # No processor's available, we'll have to skip
#                     # this frame; you may want to print a warning
#                     # here to see whether you hit this case
#                     self.processor = None
#         if self.processor:
#             self.processor.stream.write(buf)

#     def flush(self):
#         # When told to flush (this indicates end of recording), shut
#         # down in an orderly fashion. First, add the current processor
#         # back to the pool
#         if self.processor:
#             with self.lock:
#                 self.pool.append(self.processor)
#                 self.processor = None
#         # Now, empty the pool, joining each thread as we go
#         while True:
#             with self.lock:
#                 try:
#                     proc = self.pool.pop()
#                 except IndexError:
#                     pass # pool is empty
#             proc.terminated = True
#             proc.join()

# with picamera.PiCamera(resolution='VGA') as camera:
#     camera.start_preview()
#     time.sleep(2)
#     output = ProcessOutput()
#     camera.start_recording(output, format='mjpeg')
#     try:
#         while not output.done:
#             camera.wait_recording(1)
#     finally:
#         time.sleep(2)
#         print('before')
#         # camera.stop_recording()
#         camera.close()
#         print('after')