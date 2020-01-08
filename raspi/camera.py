import io
import time
import threading
import picamera
import picamera.array
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import multiprocessing
import queue
import cv2
import os
import ctypes

def print_l(print_lock, message):
    print_lock.acquire()
    try: 
        print(message)
    finally:
        print_lock.release()

def ImageProcessor(unprocessed_frames, print_lock, process_run):
    # stream = io.BytesIO()
    # self.event = multiprocessing.Event()

    while process_run:
        try:
            # attempts to get a new frame, raises exception if none available
            pass
            # print(unprocessed_frames.qsize())

            # frame_num, frame = unprocessed_frames.get_nowait()
            # print_lock.acquire()
            # print(unprocessed_frames.qsize())
            # print_lock.release()
            # stream.write(frame)
            # stream.seek(0)
            # # Read the image and do some processing on it

            # im = Image.open(stream) # read image from stream
            # img = np.array(im) # convert image to numpy array

            # # print_l(print_lock,img)
            # print_lock.acquire()
            # cv2.imwrite(f'{frame_num:03d}.png',frame)
            # # cv2.imwrite(f'{frame_num:03.3f}.png',img)
            # print_lock.release()
            # # Reset the stream
            # stream.seek(0)
            # stream.truncate()

        except queue.Empty:
            if process_run.value is False:
                break
            continue
    return

class MyAnalysisClass(picamera.array.PiRGBAnalysis):
    def __init__(self, camera):
        super(MyAnalysisClass, self).__init__(camera)
        self.number_of_processors = 4
        self.unprocessed_frames = multiprocessing.Queue()
        self.processes = []
        self.print_lock = multiprocessing.Lock()
        self.frame_num = 0
        self.start = time.time()
        self.done = False
        self.process_run = multiprocessing.Value(ctypes.c_bool,True)

        for i in range(self.number_of_processors):
            processor = multiprocessing.Process(target=ImageProcessor, args=(self.unprocessed_frames, self.print_lock, self.process_run))
            self.processes.append(processor)
            processor.start()

    def analyze(self, array):
        # print(time.time()-self.start)
        # self.start = time.time()
        if self.done is False:
            # print(self.frame_num)
            # self.unprocessed_frames.put((self.frame_num, array)) # add the new frame to the buffer
            # self.print_lock.acquire()
            # self.unprocessed_frames.put(self.frame_num) # add the new frame to the buffer
            # self.print_lock.release()
            # self.unprocessed_frames.put((time.time(), buf)) # add the new frame to the buffer
            self.frame_num += 1
            # print (time.time()-start)
            print(self.frame_num)

    def flush(self):
        print('closing camera')
        # print('flushing')
        # When told to flush (this indicates end of recording), shut
        # down in an orderly fashion.
        for p in self.processes:
            p.terminate()
            p.join()

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
#             if buf.startswith(b'\xff\xd8'): # start of a new frame
#                 # print(f'frame: {self.frame_num}')
#                 self.unprocessed_frames.put((self.frame_num, buf)) # add the new frame to the buffer
#                 # self.unprocessed_frames.put((time.time(), buf)) # add the new frame to the buffer
#                 self.frame_num += 1

#     def flush(self):
#         print('closing camera')
#         # print('flushing')
#         # When told to flush (this indicates end of recording), shut
#         # down in an orderly fashion.

#         for p in self.processes:
#             p.terminate()
#             p.join()

with picamera.PiCamera(resolution='1280x720',framerate=50) as camera:
    with picamera.array.PiRGBAnalysis(camera) as output:
        #camera.start_preview()
        camera.vflip=True
        camera.annotate_frame_num = True
        camera.annotate_text_size = 160
        camera.annotate_foreground = picamera.Color('green')
        time.sleep(2)
        output = MyAnalysisClass(camera)
        camera.start_recording(output, format='bgr')

        try:   
            camera.wait_recording(1)
        finally:
            output.process_run.value = False    # ensures all processes stop on next run
            output.done = True  # stop writing frames to queue

        while(True):
            dead_count = 0
            for process in output.processes:
                if process.is_alive() is False:
                    dead_count+=1
            if dead_count == output.number_of_processors:
                break

        camera.stop_recording()

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