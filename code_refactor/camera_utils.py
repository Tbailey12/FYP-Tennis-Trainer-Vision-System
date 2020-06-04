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
                        img = np.uint8(cv2.resize(y_data, c.RESOLUTION*c.STREAM_SCALER))
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
                        # record_flg = True
                        # os.chdir(func.make_path(root_p, c.IMG_DIR, c.RECORD_DIR))
                        # print(f"saved {n_frame_record:04d}.png")
                        # cv2.imwrite(f"{n_frame_record:04d}.png", y_data)
                        # marked_img = mark_ball_candidates(y_data, ball_candidates)
                        # cv2.imwrite(f"{n_frame_record:04d}.png", marked_img)
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
                #     print(f"{n_frame_record} frames captured")
                #     os.chdir(func.make_path(path_p))
                #     np.save("background_image.npy", background_image)
                    # record_flg = False
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
        camera.shutter_speed = c.SHUTTER_SPEED
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