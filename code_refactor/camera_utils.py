import numpy as np
import multiprocessing as mp
import queue
import cv2
import picamera
import time
import os
import io
import sys
from PIL import Image

import consts as c
import funcs as func

root_p = os.getcwd()

class BackgroundImage(object):
    def __init__(self):
        # preallocate memory for all arrays used in processing
        self.img_mean = np.zeros(c.FRAME_SIZE,dtype=np.float32)
        self.img_std = np.zeros(c.FRAME_SIZE,dtype=np.float32)

        self.mean_1 = np.zeros(c.FRAME_SIZE,dtype=np.float32)
        self.mean_2 = np.zeros(c.FRAME_SIZE,dtype=np.float32)
        self.std_1  = np.zeros(c.FRAME_SIZE,dtype=np.float32)
        self.std_2  = np.zeros(c.FRAME_SIZE,dtype=np.float32)
        self.std_3  = np.zeros(c.FRAME_SIZE,dtype=np.float32)
        self.std_4  = np.zeros(c.FRAME_SIZE,dtype=np.float32)
        self.std_5  = np.zeros(c.FRAME_SIZE,dtype=np.float32)
        self.std_6  = np.zeros(c.FRAME_SIZE,dtype=np.float32)

    def calculate_mean(self, y_data):
        # img_mean = (1 - p) * img_mean + p * y_data  # calculate mean
        np.multiply(1-c.LEARNING_RATE,  self.img_mean,  out=self.mean_1)
        np.multiply(c.LEARNING_RATE,    y_data,         out=self.mean_2)
        np.add(     self.mean_1,        self.mean_2,    out=self.img_mean)

        return True

    def calculate_std_dev(self, y_data):
        # img_std = np.sqrt((1 - p) * (img_std ** 2) + p * ((y_data - img_mean) ** 2))  # calculate std deviation
        np.square(  self.img_std,                       out=self.std_1)
        np.multiply(1-c.LEARNING_RATE,  self.std_1,     out=self.std_2)
        np.subtract(y_data,             self.img_mean,  out=self.std_3)
        np.square(  self.std_3,                         out=self.std_4)
        np.multiply(c.LEARNING_RATE,    self.std_4,     out=self.std_5)
        np.add(     self.std_2,         self.std_5,     out=self.std_6)
        np.sqrt(    self.std_6,                         out=self.img_std)

        return True

class ForegroundImage(object):
    def __init__(self):
        # preallocate memory for all arrays used in processing
        self.last_foreground = np.zeros(c.FRAME_SIZE,dtype=np.uint8)
        self.foreground = np.zeros(c.FRAME_SIZE,dtype=np.uint8)
        self.foreground_diff = np.zeros(c.FRAME_SIZE,dtype=np.uint8)

        self.A          = np.zeros(c.FRAME_SIZE,dtype=np.uint8)
        self.B_1_std    = np.zeros(c.FRAME_SIZE,dtype=np.float32)
        self.B_1_mean   = np.zeros(c.FRAME_SIZE,dtype=np.float32)
        self.B_greater  = np.zeros(c.FRAME_SIZE,dtype=np.uint8)
        self.B_2_mean   = np.zeros(c.FRAME_SIZE,dtype=np.float32)
        self.B_less     = np.zeros(c.FRAME_SIZE,dtype=np.uint8)

        self.filter_kernel = (1/16)*np.ones((4,4), dtype=np.uint8)

    def extract_foreground(self, y_data, background_image):
        self.last_foreground = np.copy(self.foreground)
        # B = np.logical_or((y_data > (img_mean + 2*img_std)),
                          # (y_data < (img_mean - 2*img_std)))  # foreground new
        np.multiply(background_image.img_std, c.FOREGROUND_SENS, out=self.B_1_std)
        np.add(     background_image.img_mean, self.B_1_std, out=self.B_1_mean)
        np.subtract(background_image.img_mean, self.B_1_std, out=self.B_2_mean)

        np.greater(   y_data, self.B_1_mean, out=self.B_greater)
        np.less(      y_data, self.B_2_mean, out=self.B_less)

        np.logical_or(self.B_greater, self.B_less, out=self.foreground)

        return True

    def foreground_difference(self):
        np.invert(np.logical_and(self.last_foreground, self.foreground), out=self.A)  # difference between prev foreground and new foreground
        np.logical_and(self.A, self.foreground, out=self.foreground_diff)   # different from previous frame and part of new frame
        self.foreground_diff = 255*self.foreground_diff.astype(np.uint8)

        return True

    def filter_foreground(self):
        self.foreground_diff = cv2.filter2D(self.foreground_diff, ddepth = -1, kernel=self.filter_kernel)
        self.foreground_diff[self.foreground_diff<c.LPF_THRESH] = 0
        self.foreground_diff[self.foreground_diff>=c.LPF_THRESH] = 255

        return True

    def get_ball_candidates(self):
        n_features_cv, labels_cv, stats_cv, centroids_cv = cv2.connectedComponentsWithStats(self.foreground_diff, connectivity=4)
        label_mask_cv = np.logical_and(stats_cv[:,cv2.CC_STAT_AREA]>c.BALL_SIZE_MIN, stats_cv[:,cv2.CC_STAT_AREA]<c.BALL_SIZE_MAX)
        ball_candidates = np.concatenate((stats_cv[label_mask_cv,2:],centroids_cv[label_mask_cv]), axis=1)

        return ball_candidates

    def sort_ball_candidates(self, ball_candidates):
        sorted_ball_candidates = ball_candidates[ball_candidates[:,c.SIZE].argsort()[::-1][:c.N_OBJECTS]]

        return sorted_ball_candidates


def mark_ball_candidates(img, ball_candidates):
    marked_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    ball_candidates = ball_candidates.astype(int)
    for ball in ball_candidates:
        cv2.drawMarker(marked_img, (ball[c.X_COORD],ball[c.Y_COORD]),(0, 0, 255),cv2.MARKER_CROSS,thickness=1,markerSize=10)
    return marked_img

def image_processor(frame_queues, event_manager, process_complete):
    processing = False
    process_complete.set()

    background_image = BackgroundImage()
    foreground_image = ForegroundImage()

    last_mean_frame = 0

    ## -- TESTING >> ##
    record_flg = False
    # filter_kernel = (1/16)*np.ones((4,4), dtype=np.uint8)
    ## << TESTING -- ##

    while True:
        time.sleep(1E-6)
        try:
            if event_manager.shutdown.is_set():
                process_complete.set()
                return

            n_frame_record, n_frame_idle, frame_buf = frame_queues.unprocessed_frames.get_nowait()
            y_data = np.frombuffer(frame_buf, dtype=np.uint8, count=c.FRAME_HEIGHT*c.FRAME_WIDTH).reshape(c.FRAME_SIZE).astype(np.float32)

            # print(n_frame_record, n_frame_idle)
            # print(frame_queues.unprocessed_frames.qsize())

            if event_manager.recording.is_set() or not process_complete.is_set():

                # streaming
                if event_manager.record_stream.is_set():
                    if n_frame_record%c.STREAM_IMG_DELTA == 0 and n_frame_record >= 0:
                        img = np.uint8(cv2.resize(y_data, (320,240)))
                        frame_queues.processed_frames.put((n_frame_record, img))

                # recording
                else:
                    if n_frame_record >= 0:
                        ## -- TESTING >> ##
                        # y_data = cv2.filter2D(y_data, ddepth = -1, kernel=filter_kernel)
                        ## << TESTING -- ##

                        foreground_image.extract_foreground(y_data, background_image)
                        foreground_image.foreground_difference()
                        foreground_image.filter_foreground()
                        ball_candidates = foreground_image.get_ball_candidates()
                        ball_candidates = foreground_image.sort_ball_candidates(ball_candidates)
                        frame_queues.processed_frames.put((n_frame_record, ball_candidates))

                        ## -- TESTING >> ##
                        record_flg = True
                        os.chdir(func.make_path(root_p, c.IMG_DIR, c.RECORD_DIR))
                        print(f"saved {n_frame_record:04d}.png")
                        # cv2.imwrite(f"{n_frame_record:04d}.png", y_data)
                        marked_img = mark_ball_candidates(y_data, ball_candidates)
                        cv2.imwrite(f"{n_frame_record:04d}.png", marked_img)
                        ## << TESTING -- ##

            # elif not event_manager.recording.is_set() and n_frame_record == -1:
            #     process_complete.set()
            # calculate mean and standard deviation while idle
            else:
                if n_frame_idle > (last_mean_frame + c.BACKGROUND_RATE):
                    background_image.calculate_mean(y_data)
                    background_image.calculate_std_dev(y_data)
                    last_mean_frame = n_frame_idle 
                    # print(f"mean: {background_image.img_mean.mean()}")
                    # print(f"std: {background_image.img_std.mean()}")      
            
        except queue.Empty:
            # print(frame_queues.unprocessed_frames.qsize())

            if not event_manager.recording.is_set() and frame_queues.unprocessed_frames.qsize() == 0:  # if the recording has finished
                ## -- TESTING >> ##
                # if record_flg:
                #     os.chdir(func.make_path(path_p))
                #     np.save("background_image.npy", background_image)
                #     record_flg = False
                ## << TESTING -- ##
                process_complete.set()
        
class EventManager(object):
    '''
    Manages an array of mp.Events shared between processes
    '''
    def __init__(self):
        self.picam_ready = mp.Event()
        self.recording = mp.Event()
        self.record_stream = mp.Event()
        self.processing_complete = mp.Event()
        self.shutdown = mp.Event()
        
class FrameController(object):
    def __init__(self, frame_queues, event_manager, processing_complete_list):
        self.frame_queues = frame_queues
        self.event_manager = event_manager
        self.processing_complete_list = processing_complete_list

        self.processor_list = []
        self.n_frame_record = 0
        self.n_frame_idle = 0

        self.stream = io.BytesIO()

        # create processors for processing the individual video frames
        for i in range(c.N_PROCESSORS):
            proccess_complete = mp.Event()
            processor = mp.Process(target=image_processor, args=(self.frame_queues, self.event_manager, proccess_complete))
            self.processing_complete_list.append(proccess_complete)
            self.processor_list.append(processor)
            processor.start()

    def write(self, buf):
        '''
        Called whenever there is a new frame available
        n_frame_record keeps track of the frame number since recording
        n_frame_idle keep track of the frame number since camera initialisation
        '''
        if self.event_manager.recording.is_set():
            self.frame_queues.unprocessed_frames.put((self.n_frame_record, self.n_frame_idle, buf))

        elif self.event_manager.processing_complete.is_set() and self.n_frame_idle%c.BACKGROUND_RATE == 0:
            for i in range(c.N_PROCESSORS):
                self.frame_queues.unprocessed_frames.put((self.n_frame_record, self.n_frame_idle, buf))

        if self.event_manager.recording.is_set() or self.event_manager.record_stream.is_set():
            self.n_frame_record += 1
            self.n_frame_idle += 1
        else:
            self.n_frame_record = -1
            self.n_frame_idle += 1

    def flush(self):
        '''
        At end of the recording, wait for all processes to complete
        '''
        for i, processor in enumerate(self.processor_list):
            self.processing_complete_list[i].wait()
            processor.terminate()
            processor.join()
        print('Shutdown (picam)')
        self.processing_complete.set()

        return True

class FrameQueues(object):
    def __init__(self):
        self.unprocessed_frames = mp.Queue()
        self.processed_frames = mp.Queue()

    def empty_processed(self):
        while True:
            try:
                self.processed_frames.get_nowait()
            except queue.Empty:
                break

    def empty_unprocessed(self):
        while True:
            try:
                self.unprocessed_frames.get_nowait()
            except queue.Empty:
                break

class CameraManager(object):
    def __init__(self):
        self.frame_queues = FrameQueues()
        self.event_manager = EventManager()

        self.processing_complete_list = []
        self.shutdown = mp.Event()
        self.record_t = mp.Value('f', c.REC_T)

    def add_annotations(self, camera):
        camera.annotate_frame_num = True
        camera.annotate_text_size = 160
        camera.annotate_foreground = picamera.Color('green')

        return True

    def set_params(self, camera):
        '''
        Sets all camera variables based on settings in consts.py

        Return: True if set successfully
        '''
        camera.framerate = c.FRAMERATE
        camera.resolution = c.RESOLUTION
        camera.shutter_speed = 1000
        # camera.iso = c.ISO
        camera.vflip = True
        camera.hflip = True
        camera.saturation = 100
        # self.add_annotations(camera)
        # camera.video_denoise = True

        time.sleep(2)   # give the camera a couple of seconds to initialise
        camera.exposure_mode = 'off'
        # camera.shutter_speed = camera.exposure_speed
        g = camera.awb_gains
        camera.awb_mode = 'off'
        camera.awb_gains = g

        return True

    def record(self, record_t = None):
        self.event_manager.processing_complete.wait()
        self.event_manager.processing_complete.clear()
        self.frame_queues.empty_unprocessed()
        self.frame_queues.empty_processed()

        if record_t is not None and isinstance(record_t, float):
            if record_t > 0 and record_t <= c.REC_T_MAX:
                self.record_t.value = record_t

        self.event_manager.recording.set()

    def stream(self, stream_t = None):
        self.event_manager.processing_complete.wait()
        self.event_manager.processing_complete.clear()

        if stream_t is not None and isinstance(stream_t, float):
            if stream_t > 0 and stream_t <= c.STREAM_T_MAX:
                self.record_t.value = stream_t

        # self.event_manager.processing_complete.wait()
        # self.frame_queues.empty_unprocessed()
        # self.frame_queues.empty_processed()

        self.event_manager.record_stream.set()
        self.event_manager.recording.set()

    def manage_recording(self, camera):
        '''
        Manages the main recording loop of the camera
        '''
        # Picam will stay in this while loop waiting for either shutdown or recording flags to be set
        while True:
            if self.event_manager.shutdown.is_set():
                break

            elif self.event_manager.recording.is_set():
                try:
                    for process in self.processing_complete_list:  # reset all processing complete events
                        process.clear()
                    camera.wait_recording(self.record_t.value) # record for an amount of time
                finally:
                    self.event_manager.recording.clear()
                    for process in self.processing_complete_list:  # wait for all processing to be complete
                        process.wait()
                    self.event_manager.processing_complete.set()   # set processing complete event
                    self.event_manager.record_stream.clear()

        camera.stop_recording()
        return True

    def start_camera(self):
        '''
        Manages camera startup and recording

        Return: True
        '''

        with picamera.PiCamera() as camera:
            self.set_params(camera)
            output = FrameController(self.frame_queues, self.event_manager, self.processing_complete_list)
            camera.start_recording(output, format='yuv')
            self.event_manager.picam_ready.set()
            self.manage_recording(camera)

        self.shutdown.set()

# ## -- imports -- ##
# import io
# import time
# import traceback
# import picamera
# import numpy as np
# from PIL import Image
# import multiprocessing as mp
# import queue
# import cv2
# import os
# import sys
# import RPi.GPIO as GPIO
# from skimage import measure
# from scipy import ndimage

# ## -- custom imports -- ##
# import client
# import consts as c
# import socket_funcs as sf
# from l_r_consts import *

# ## -- settings -- ##
# w,h = c.RESOLUTION
# client_name = CLIENT_NAME

# kernel4 = (1/16)*np.ones((4,4), dtype=np.uint8)

# class EventManager(object):
#     def __init__(self, event_change, recording, record_stream, capture_img, shutdown):
#         self.event_change = event_change
#         self.recording = recording
#         self.record_stream = record_stream
#         self.capture_img = capture_img
#         self.shutdown = shutdown
#         self.events = [self.recording, self.record_stream, self.capture_img]

#     def update(self, flag = None):
#         if not self.event_change.is_set():
#             self.event_change.set()
#             for event in self.events:
#                 if event is flag:
#                     event.set()
#                     break

#     def clear(self):
#         self.event_change.clear()
#         for event in self.events:
#             event.clear()

# def ImageProcessor(unprocessed_frames, processed_frames, proc_complete, event_manager, n_calib):
#     processing = False
#     proc_complete.set()

#     # preallocate memory for all arrays used in processing
#     img_mean = np.zeros([h,w],dtype=np.float32)
#     img_std = np.ones([h,w],dtype=np.float32)

#     A =  np.zeros([h,w],dtype=np.uint8)
#     B = np.zeros([h,w],dtype=np.uint8)
#     B_old = np.zeros([h,w],dtype=np.uint8)
#     C = np.zeros([h,w],dtype=np.uint8)

#     mean_1 = np.zeros([h,w],dtype=np.float32)
#     mean_2 = np.zeros([h,w],dtype=np.float32)

#     std_1 = np.zeros([h,w],dtype=np.float32)
#     std_2 = np.zeros([h,w],dtype=np.float32)
#     std_3 = np.zeros([h,w],dtype=np.float32)
#     std_4 = np.zeros([h,w],dtype=np.float32)
#     std_5 = np.zeros([h,w],dtype=np.float32)
#     std_6 = np.zeros([h,w],dtype=np.float32)

#     B_1_std = np.zeros([h,w],dtype=np.float32)
#     B_1_mean = np.zeros([h,w],dtype=np.float32)
#     B_greater = np.zeros([h,w],dtype=np.uint8)
#     B_2_mean = np.zeros([h,w],dtype=np.float32)
#     B_less = np.zeros([h,w],dtype=np.uint8)

#     ########## FOR TESTING ##############
#     img_array = []
#     C_array = []
#     # std_data = np.zeros([h,w],dtype=np.float32)
#     # mean_data = np.zeros([h,w],dtype=np.float32)
#     save_arr = False
#     y_data = np.zeros((h,w))
#     img_temp = np.zeros((3,h,w))
#     C_temp = np.zeros((3,h,w))
#     temp_img = np.zeros((h,w))
#     ########## FOR TESTING ##############

#     last_n_frame = 0

#     p = c.LEARNING_RATE

#     time_cumulative = 0

#     total_time = 0
#     total_frames = 0

#     while True:
#         time.sleep(1E-4)
#         try:
#             if event_manager.shutdown.is_set():
#                 proc_complete.set()
#                 return
#             # get the frames from queue
#             n_frame, n_frame_2, frame_buf = unprocessed_frames.get_nowait()
            
#             # y_data is a numpy array hxw with 8 bit greyscale brightness values
#             y_data = np.frombuffer(frame_buf, dtype=np.uint8, count=w*h).reshape((h,w)).astype(np.float32)

#             # every 3 frames update the background
#             if n_frame_2 > (last_n_frame+c.BACKGROUND_RATE) and not event_manager.event_change.is_set():
#                 last_n_frame = n_frame_2

#                 # img_mean = (1 - p) * img_mean + p * y_data  # calculate mean
#                 np.multiply(1-p,img_mean,out=mean_1)
#                 np.multiply(p,y_data,out=mean_2)
#                 np.add(mean_1,mean_2,out=img_mean)
                
#                 # img_std = np.sqrt((1 - p) * (img_std ** 2) + p * ((y_data - img_mean) ** 2))  # calculate std deviation
#                 np.square(img_std,out=std_1)
#                 np.multiply(1-p,std_1,out=std_2)
#                 np.subtract(y_data,img_mean,out=std_3)
#                 np.square(std_3,out=std_4)
#                 np.multiply(p,std_4,out=std_5)
#                 np.add(std_2,std_5,out=std_6)
#                 np.sqrt(std_6,out=img_std)

#             if not proc_complete.is_set() and n_frame > -1:
#                 mean_data = img_mean
#                 std_data = img_std

#                 start = time.time_ns()
#                 B_old = np.copy(B)
#                 # B = np.logical_or((y_data > (img_mean + 2*img_std)),
#                                   # (y_data < (img_mean - 2*img_std)))  # foreground new
#                 np.multiply(img_std,3,out=B_1_std)
#                 np.add(B_1_std,img_mean,out=B_1_mean)
#                 B_greater = np.greater(y_data,B_1_mean)
#                 np.subtract(img_mean,B_1_std,out=B_2_mean)
#                 B_less = np.less(y_data,B_2_mean)
#                 B = np.logical_or(B_greater,B_less)

#                 A = np.invert(np.logical_and(B_old, B))  # difference between prev foreground and new foreground
#                 C = np.logical_and(A, B)   # different from previous frame and part of new frame
                
#                 C = 255*C.astype(np.uint8)

#                 C = cv2.filter2D(C, ddepth = -1, kernel=kernel4)
#                 C[C<c.LPF_THRESH] = 0
#                 C[C>=c.LPF_THRESH] = 255

#                 n_features_cv, labels_cv, stats_cv, centroids_cv = cv2.connectedComponentsWithStats(C, connectivity=4)

#                 label_mask_cv = np.logical_and(stats_cv[:,cv2.CC_STAT_AREA]>2, stats_cv[:,cv2.CC_STAT_AREA]<10000)
#                 ball_candidates = np.concatenate((stats_cv[label_mask_cv,2:],centroids_cv[label_mask_cv]), axis=1)

#                 # sort ball candidates by size and keep the top 100
#                 ball_candidates = ball_candidates[ball_candidates[:,c.SIZE].argsort()[::-1][:c.N_OBJECTS]]

#                 processed_frames.put((n_frame, ball_candidates))

#                 ########## FOR TESTING ##############
#                 C_temp = cv2.cvtColor(C, cv2.COLOR_GRAY2RGB)
#                 img_temp = cv2.cvtColor(y_data, cv2.COLOR_GRAY2RGB)
#                 ball_candidates = ball_candidates.astype(int)
#                 for ball in ball_candidates:
#                     cv2.drawMarker(C_temp,(ball[c.X_COORD],ball[c.Y_COORD]),(0, 0, 255),cv2.MARKER_CROSS,thickness=2,markerSize=10)
#                     cv2.drawMarker(img_temp,(ball[c.X_COORD],ball[c.Y_COORD]),(0, 0, 255),cv2.MARKER_CROSS,thickness=2,markerSize=10)
                
#                 save_arr = True
#                 img_array.append((n_frame, y_data))
#                 C_array.append(C_temp)

#                 total_time += (time.time_ns()-start)
#                 total_frames += 1
#                 ########## FOR TESTING ##############

#             elif event_manager.record_stream.is_set() and n_frame > -1:
#                 if n_frame%n_calib.value == 0:
#                     processed_frames.put((n_frame, y_data))
            
#         except queue.Empty:
#                 if not event_manager.recording.is_set() and unprocessed_frames.qsize() == 0:  # if the recording has finished
#                     proc_complete.set()     # set the proc_complete event
#                     ######### FOR TESTING ##############
#                     if total_frames>0:
#                         print((total_time/total_frames)/1E6)
#                         total_frames = 0
#                         total_time = 0
#                     if img_array is not None and save_arr:
#                         for i,img in enumerate(img_array):
#                             frame, data = img
#                             os.chdir(c.IMG_P)
#                             cv2.imwrite(f"{frame:04d}.png",data)
#                             cv2.imwrite(f"C{frame:04d}.png",C_array[i])
#                         os.chdir(c.DATA_P)
#                         np.save('img_mean',mean_data)
#                         np.save('img_std', std_data)
#                         os.chdir(c.ROOT_P)
#                         img_array = []
#                         C_array = []
#                         save_arr = False
#                     ######### FOR TESTING ##############



# def StartPicam(unprocessed_frames, processed_frames, picam_ready, processing_complete, t_record, n_calib, event_manager):
#     proc_complete = []     # contains processing events, true if complete
#     with picamera.PiCamera() as camera:
#         camera.framerate = c.FRAMERATE
#         camera.resolution = c.RESOLUTION
#         camera.iso = c.ISO
#         camera.vflip = True # camera is upside down so flip the image
#         camera.hflip = True # flip horizontal too
#         ## -- adds the frame number to the image for testing
#         # camera.annotate_frame_num = True
#         camera.annotate_text_size = 160
#         camera.annotate_foreground = picamera.Color('green')
#         ## ----------------------------------------------- ##
#         time.sleep(2)   # give the camera a couple of seconds to initialise
#         camera.shutter_speed = camera.exposure_speed
#         camera.exposure_mode = 'off'
#         g = camera.awb_gains
#         camera.awb_mode = 'off'
#         camera.awb_gains = g


#         print(f"shutter: {camera.shutter_speed}")
#         print(f"awb: {g}")
#         print(f'iso: {camera.iso}')


#         output = FrameController(unprocessed_frames, processed_frames, proc_complete, n_calib, event_manager)
#         camera.start_recording(output, format='yuv')

#         picam_ready.set()   # the picam is ready to record

#         # Picam will stay in this while loop waiting for either shutdown or recording flags to be set
#         while True:
#             if event_manager.event_change.wait():
#                 if event_manager.recording.is_set():
#                     try:
#                         for proc in proc_complete:  # reset all processing complete events
#                             proc.clear()
#                         camera.wait_recording(t_record.value) # record for an amount of time
#                     finally:
#                         event_manager.recording.clear()
#                         for proc in proc_complete:  # wait for all processing to be complete
#                             proc.wait()
#                         processing_complete.set()   # set processing complete event

#                 elif event_manager.record_stream.is_set():
#                     processing_complete.wait()

#                 elif event_manager.capture_img.is_set():
#                     processing_complete.wait()

#                 elif event_manager.shutdown.is_set():
#                     break
#         camera.stop_recording()
#     return


# class LED(object):
#     def __init__(self, led_pin, led_freq, shutdown):
#         self.led_pin = led_pin
#         self.led_freq = led_freq
#         self.shutdown = shutdown
#         self.process = None

#         GPIO.setmode(GPIO.BCM)
#         GPIO.setwarnings(False)
#         GPIO.setup(led_pin, GPIO.OUT)

#     def Update(self, led_freq, force = False):
#         if self.led_freq == led_freq and force == False:
#             return
#         self.led_freq = led_freq
#         # terminate any existing LED process
#         if self.process:
#             self.process.terminate()
#             self.process.join()
#         # start new LED process with the new flash frequency
#         self.process = mp.Process(target=self.Run,args=(self.led_pin,self.led_freq,self.shutdown))
#         self.process.start()

#     def Kill(self):
#         if self.process is not None:
#             self.process.terminate()
#             self.process.join()
#         GPIO.cleanup()

#     def Run(self,led_pin,led_freq,shutdown):
#         while True:
#             t = 0
#             if (led_freq >= c.LED_F_MIN) and (led_freq <= c.LED_F_MAX):
#                 t = abs(1/(2*led_freq))

#                 GPIO.output(led_pin, GPIO.HIGH)
#                 time.sleep(t)
#                 GPIO.output(led_pin, GPIO.LOW)
#                 time.sleep(t)
#             elif led_freq > c.LED_F_MAX:
#                 GPIO.output(led_pin, GPIO.HIGH)
#                 time.sleep(1)
#             else:
#                 GPIO.output(led_pin, GPIO.LOW)
#                 time.sleep(1)

# def record(r_led, g_led, event_manager, processing_complete, processed_frames):
#     print('recording')
#     g_led.Update(100, force=True)
#     r_led.Update(1, force=True)
#     processing_complete.wait()
#     processing_complete.clear()
#     event_manager.update(recording)

#     errors = 0
#     while True: # wait here until all frames have been processed and sent to server
#         try: 
#             # get the processed frames from queue
#             n_frame, ball_candidates = processed_frames.get_nowait()
#             # print(f"rec: {n_frame}")
#             message = sf.MyMessage(c.TYPE_BALLS, (n_frame, ball_candidates))
#             if not sf.send_message(client.client_socket, message, c.CLIENT):
#                 errors += 1
#                 print(f"error: {errors}")
        
#         # if processing complete and no more data to send to server
#         except queue.Empty:
#             if processing_complete.is_set() and (processed_frames.qsize() == 0):
#                 event_manager.clear()
#                 break
#     # if there were no transmission errors send True, else send False
#     if errors == 0:
#         message = sf.MyMessage(c.TYPE_DONE, True)
#     else:
#         message = sf.MyMessage(c.TYPE_DONE, False)
#     sf.send_message(client.client_socket, message, c.CLIENT)
#     return

# def stream(r_led, g_led, event_manager, processing_complete, processed_frames):
#     print('recording stream')
#     message_list = []
#     g_led.Update(1, force=True)
#     r_led.Update(1, force=True)
#     processing_complete.wait()
#     processing_complete.clear()
#     event_manager.update(record_stream)
#     errors = 0

#     while True:
#         try:
#             # get the frames from queue
#             n_frame, frame_buf = processed_frames.get_nowait()
            
#             if event_manager.record_stream.is_set():
#                 print(f"calib: {n_frame}")
#                 # y_data is a numpy array hxw with 8 bit greyscale brightness values
#                 y_data = np.frombuffer(frame_buf, dtype=np.float32, count=w*h).reshape((h,w)).astype(np.uint8)
#                 message = sf.MyMessage(c.TYPE_IMG, (n_frame, y_data))
#                 if not sf.send_message(client.client_socket, message, c.CLIENT):
#                     errors += 1
#                     print(f"error: {errors}")

#                 message_list.extend(client.read_all_server_messages())
#                 for message in message_list:
#                     if message['data'].type == c.TYPE_DONE:
#                         event_manager.clear()

#         except queue.Empty:
#             if not event_manager.record_stream.is_set() and processed_frames.qsize() == 0:  # if the recording has finished
#                 print('stream done')
#                 processing_complete.set()
#                 break

#     # if there were no transmission errors send True, else send False
#     if errors == 0:
#         message = sf.MyMessage(c.TYPE_DONE, True)
#     else:
#         message = sf.MyMessage(c.TYPE_DONE, False)

#     sf.send_message(client.client_socket, message, c.CLIENT)



# def shutdown_prog(r_led, g_led, event_manager):
#     r_led.Kill()
#     g_led.Kill()
#     event_manager.update()
#     event_manager.shutdown.set()
#     print('shutdown (main)')
#     return


# if __name__ == "__main__":
#     message_list = []

#     ## -- setup client connection to server -- ##

#     client.connect_to_server(name=client_name) 

#     ## -- initialise multithreading objs -- ##
#     unprocessed_frames = mp.Queue()
#     processed_frames = mp.Queue()
#     shutdown = mp.Event()
#     picam_ready = mp.Event()
#     processing_complete = mp.Event()

#     event_change = mp.Event()
#     recording = mp.Event()
#     record_stream = mp.Event()
#     capture_img = mp.Event()

#     event_manager = EventManager(event_change, recording, record_stream, capture_img, shutdown)

#     t_record = mp.Value('i',c.REC_T)
#     n_calib = mp.Value('i', int(c.FRAMERATE*c.CALIB_IMG_DELAY))

#     r_led = LED(c.R_LED_PIN, 0, shutdown)
#     g_led = LED(c.G_LED_PIN, 0, shutdown)

#     processing_complete.set() # inintially there is no processing being done

#     ## -- initialise Picam process for recording
#     Picam = mp.Process(target=StartPicam, args=(unprocessed_frames, processed_frames, picam_ready, processing_complete, t_record, n_calib, event_manager))
#     Picam.start()
#     picam_ready.wait()

#     while True and not shutdown.is_set():
#         try:
#             message_list = []
#             r_led.Update(0)
#             g_led.Update(1)
#             ## -- read server messages -- ##
#             message_list.extend(client.read_all_server_messages())

#             for message in message_list:
#                 try:
#                     # record message
#                     if message['data'].type == c.TYPE_REC:
#                         # change the recording time if the new recording time is valid
#                         if isinstance(message['data'].message, int):
#                             t_record.value = message['data'].message
#                         else:
#                             t_record.value = c.REC_T
#                         record(r_led, g_led, event_manager, processing_complete, processed_frames)
#                         break
#                     elif message['data'].type == c.TYPE_STREAM:
#                         n_calib.value = int(c.FRAMERATE*message['data'].message)
#                         stream(r_led, g_led, event_manager, processing_complete, processed_frames)
#                         break

#                     elif message['data'].type == c.TYPE_SHUTDOWN:
#                         processing_complete.clear()
#                         shutdown_prog(r_led, g_led, event_manager)
#                         processing_complete.wait()
#                         Picam.terminate()
#                         Picam.join()
#                         print('shutdown complete')
#                         break

#                 except Exception as e:
#                     print(e)
#                     print('unrecognised message type')
#                     continue

#         except sf.CommError as e:
#             traceback.print_exc(file=sys.stdout)
#             processing_complete.clear()
#             shutdown_prog(r_led, g_led, event_manager)
#             processing_complete.wait()
#             Picam.terminate()
#             Picam.join()
#             print('shutdown complete')
#             break