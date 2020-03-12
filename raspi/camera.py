## -- imports -- ##
import io
import time
import traceback
import picamera
import numpy as np
from PIL import Image
import multiprocessing as mp
import queue
import cv2
import os
import sys
import RPi.GPIO as GPIO
from skimage import measure
from scipy import ndimage

## -- custom imports -- ##
import client
import consts as c
import socket_funcs as sf
from l_r_consts import *

## -- settings -- ##
w,h = c.RESOLUTION
client_name = CLIENT_NAME

class EventManager(object):
    def __init__(self, event_change, recording, record_stream, capture_img, shutdown):
        self.event_change = event_change
        self.recording = recording
        self.record_stream = record_stream
        self.capture_img = capture_img
        self.shutdown = shutdown
        self.events = [self.recording, self.record_stream, self.capture_img]

    def update(self, flag = None):
        if not self.event_change.is_set():
            self.event_change.set()
            for event in self.events:
                if event is flag:
                    event.set()
                    break

    def clear(self):
        self.event_change.clear()
        for event in self.events:
            event.clear()

def ImageProcessor(unprocessed_frames, processed_frames, proc_complete, event_manager, n_calib):
    processing = False
    proc_complete.set()

    # preallocate memory for all arrays used in processing
    img_mean = np.zeros([h,w],dtype=np.float32)
    img_std = np.ones([h,w],dtype=np.float32)

    A =  np.zeros([h,w],dtype=np.uint8)
    B = np.zeros([h,w],dtype=np.uint8)
    B_old = np.zeros([h,w],dtype=np.uint8)
    C = np.zeros([h,w],dtype=np.uint8)

    mean_1 = np.zeros([h,w],dtype=np.float32)
    mean_2 = np.zeros([h,w],dtype=np.float32)

    std_1 = np.zeros([h,w],dtype=np.float32)
    std_2 = np.zeros([h,w],dtype=np.float32)
    std_3 = np.zeros([h,w],dtype=np.float32)
    std_4 = np.zeros([h,w],dtype=np.float32)
    std_5 = np.zeros([h,w],dtype=np.float32)
    std_6 = np.zeros([h,w],dtype=np.float32)

    B_1_std = np.zeros([h,w],dtype=np.float32)
    B_1_mean = np.zeros([h,w],dtype=np.float32)
    B_greater = np.zeros([h,w],dtype=np.uint8)
    B_2_mean = np.zeros([h,w],dtype=np.float32)
    B_less = np.zeros([h,w],dtype=np.uint8)

    ########## FOR TESTING ##############
    img_array = np.zeros((275,h,w))
    std_data = np.zeros([h,w],dtype=np.float32)
    mean_data = np.zeros([h,w],dtype=np.float32)
    save_arr = False
    # y_data = np.zeros((h,w))
    # temp_img = np.zeros((h,w))
    ########## FOR TESTING ##############

    # kernel = np.ones((2,2), dtype = np.uint8)
    # kernel2 = np.array( [[0,0,1,0,0],
    #                     [0,1,1,1,0],
    #                     [1,1,1,1,1],
    #                     [0,1,1,1,0],
    #                     [0,0,1,0,0]], dtype = np.uint8)

    last_n_frame = 0

    p = c.LEARNING_RATE

    time_cumulative = 0

    while True:
        try:
            if event_manager.shutdown.is_set():
                proc_complete.set()
                return
            # get the frames from queue
            n_frame, n_frame_2, frame_buf = unprocessed_frames.get_nowait()
            # print(f"{n_frame}:{n_frame_2}")
            
            # y_data is a numpy array hxw with 8 bit greyscale brightness values
            y_data = np.frombuffer(frame_buf, dtype=np.uint8, count=w*h).reshape((h,w)).astype(np.float32)

            # every 3 frames update the background
            if n_frame_2 > (last_n_frame+c.BACKGROUND_RATE) and not event_manager.event_change.is_set():
                last_n_frame = n_frame_2

                # img_mean = (1 - p) * img_mean + p * y_data  # calculate mean
                np.multiply(1-p,img_mean,out=mean_1)
                np.multiply(p,y_data,out=mean_2)
                np.add(mean_1,mean_2,out=img_mean)
                
                # img_std = np.sqrt((1 - p) * (img_std ** 2) + p * ((y_data - img_mean) ** 2))  # calculate std deviation
                np.square(img_std,out=std_1)
                np.multiply(1-p,std_1,out=std_2)
                np.subtract(y_data,img_mean,out=std_3)
                np.square(std_3,out=std_4)
                np.multiply(p,std_4,out=std_5)
                np.add(std_2,std_5,out=std_6)
                np.sqrt(std_6,out=img_std)

            if not proc_complete.is_set() and n_frame > -1:
                ########## FOR TESTING ##############
                # start = time.time_ns()
                # temp_img = cv2.boxFilter(y_data,ddepth=-1,ksize=(5,5),anchor=(-1,-1),normalize=True)
                # print((time.time_ns()-start)/1E6)
                std_data = img_std
                mean_data = img_mean
                ########## FOR TESTING ##############

                B_old = np.copy(B)
                # B = np.logical_or((y_data > (img_mean + 2*img_std)),
                                  # (y_data < (img_mean - 2*img_std)))  # foreground new
                np.multiply(img_std,2,out=B_1_std)
                np.add(B_1_std,img_mean,out=B_1_mean)
                B_greater = np.greater(y_data,B_1_mean)
                np.subtract(img_mean,B_1_std,out=B_2_mean)
                B_less = np.less(y_data,B_2_mean)
                B = np.logical_or(B_greater,B_less)

                A = np.invert(np.logical_and(B_old, B))  # difference between prev foreground and new foreground
                C = np.logical_and(A, B)   # different from previous frame and part of new frame
                C = 255*C.astype(np.uint8)

                ########## FOR TESTING ##############
                C = np.zeros([h,w],dtype=np.uint8)
                save_arr = True
                img_array[n_frame] = y_data
                # C_array[n_frame] = C
                # cv2.imwrite(f"{n_frame:04d}.png",y_data)
                ########## FOR TESTING ##############

                n_features_cv, labels_cv, stats_cv, centroids_cv = cv2.connectedComponentsWithStats(C, connectivity=8)

                label_mask_cv = np.logical_and(stats_cv[:,cv2.CC_STAT_AREA]>1, stats_cv[:,cv2.CC_STAT_AREA]<100)
                ball_candidates = np.concatenate((stats_cv[label_mask_cv],centroids_cv[label_mask_cv]), axis=1)

                # print((time.time_ns()-start)/1E6)
                processed_frames.put((n_frame, ball_candidates))

            elif event_manager.record_stream.is_set() and n_frame > -1:
                if n_frame%n_calib.value == 0:
                    processed_frames.put((n_frame, y_data))
                    # cv2.imwrite(f"{n_frame:04d}.png",y_data)
            
        except queue.Empty:
                if not event_manager.recording.is_set() and unprocessed_frames.qsize() == 0:  # if the recording has finished
                    proc_complete.set()     # set the proc_complete event
                    ######### FOR TESTING ##############
                    if img_array is not None and save_arr:
                        for i,img in enumerate(img_array):
                            if np.mean(img) > 0:
                                os.chdir(c.IMG_P)
                                cv2.imwrite(f"{i:04d}.png",img)
                                os.chdir(c.ROOT_P)
                                # cv2.imwrite(f"C{i:04d}.png",C_array[i])
                        os.chdir(c.DATA_P)
                        np.save('img_mean',mean_data)
                        np.save('img_std', std_data)
                        os.chdir(c.ROOT_P)
                        img_array = np.zeros((275,h,w))
                        save_arr = False
                    ######### FOR TESTING ##############


class FrameController(object):
    def __init__(self, unprocessed_frames, processed_frames, proc_complete, n_calib, event_manager):
        self.unprocessed_frames = unprocessed_frames
        self.processed_frames = processed_frames
        self.n_calib = n_calib
        self.event_manager = event_manager
        self.proc_complete = proc_complete
        self.number_of_processors = c.N_PROCESSORS
        self.processors = []
        self.n_frame = 0
        self.n_frame_2 = 0

        ## -- create processors for processing the individual video frames
        for i in range(c.N_PROCESSORS):
            proc_event = mp.Event()
            processor = mp.Process(target=ImageProcessor, args=(self.unprocessed_frames,self.processed_frames, proc_event, self.event_manager,self.n_calib))
            self.proc_complete.append(proc_event)
            self.processors.append(processor)
            processor.start()

    def write(self, buf):
        self.unprocessed_frames.put((self.n_frame, self.n_frame_2, buf))    # add the new frame to the queue
        if self.event_manager.recording.is_set() or self.event_manager.record_stream.is_set():
            self.n_frame += 1
            self.n_frame_2 += 1
        else:
            self.n_frame = -1
            self.n_frame_2 += 1

    def flush(self):
        for i, processor in enumerate(self.processors):
            self.proc_complete[i].wait() # wait until process has finished
            processor.terminate()
            processor.join()
        print('shutdown (picam)')
        processing_complete.set()
        return


def StartPicam(unprocessed_frames, processed_frames, picam_ready, processing_complete, t_record, n_calib, event_manager):
    proc_complete = []     # contains processing events, true if complete
    with picamera.PiCamera() as camera:
        camera.framerate = c.FRAMERATE
        camera.resolution = c.RESOLUTION
        camera.vflip = True # camera is upside down so flip the image
        camera.hflip = True # flip horizontal too
        ## -- adds the frame number to the image for testing
        # camera.annotate_frame_num = True
        camera.annotate_text_size = 160
        camera.annotate_foreground = picamera.Color('green')
        ## ----------------------------------------------- ##
        time.sleep(2)   # give the camera a couple of seconds to initialise
        camera.shutter_speed = camera.exposure_speed
        camera.exposure_mode = 'off'
        g = camera.awb_gains
        camera.awb_mode = 'off'
        camera.awb_gains = g


        output = FrameController(unprocessed_frames, processed_frames, proc_complete, n_calib, event_manager)
        camera.start_recording(output, format='yuv')

        picam_ready.set()   # the picam is ready to record

        # Picam will stay in this while loop waiting for either shutdown or recording flags to be set
        while True:
            if event_manager.event_change.wait():
                if event_manager.recording.is_set():
                    try:
                        for proc in proc_complete:  # reset all processing complete events
                            proc.clear()
                        camera.wait_recording(t_record.value) # record for an amount of time
                    finally:
                        event_manager.recording.clear()
                        for proc in proc_complete:  # wait for all processing to be complete
                            proc.wait()
                        processing_complete.set()   # set processing complete event

                elif event_manager.record_stream.is_set():
                    processing_complete.wait()

                elif event_manager.capture_img.is_set():
                    processing_complete.wait()

                elif event_manager.shutdown.is_set():
                    break
        camera.stop_recording()
    return
    # return

            # if recording.wait(1):    # wait for recording flag to be set in main()
            #     processing_complete.clear()
            #     try:
            #         for proc in proc_complete:  # reset all processing complete events
            #             proc.clear()
            #         camera.wait_recording(t_record.value) # record for an amount of time
            #     finally:
            #         recording.clear()   # clear the record flag to stop processing
            #         for proc in proc_complete:  # wait for all processing to be complete
            #             proc.wait()
            #         processing_complete.set()   # set processing complete event
            # elif shutdown.is_set():
            #     print('shutdown (picam)')
            #     break 


class LED(object):
    def __init__(self, led_pin, led_freq, shutdown):
        self.led_pin = led_pin
        self.led_freq = led_freq
        self.shutdown = shutdown
        self.process = None

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(led_pin, GPIO.OUT)

    def Update(self, led_freq, force = False):
        if self.led_freq == led_freq and force == False:
            return
        self.led_freq = led_freq
        # terminate any existing LED process
        if self.process:
            self.process.terminate()
            self.process.join()
        # start new LED process with the new flash frequency
        self.process = mp.Process(target=self.Run,args=(self.led_pin,self.led_freq,self.shutdown))
        self.process.start()

    def Kill(self):
        if self.process is not None:
            self.process.terminate()
            self.process.join()
        GPIO.cleanup()

    def Run(self,led_pin,led_freq,shutdown):
        while True:
            t = 0
            if (led_freq >= c.LED_F_MIN) and (led_freq <= c.LED_F_MAX):
                t = abs(1/(2*led_freq))

                GPIO.output(led_pin, GPIO.HIGH)
                time.sleep(t)
                GPIO.output(led_pin, GPIO.LOW)
                time.sleep(t)
            elif led_freq > c.LED_F_MAX:
                GPIO.output(led_pin, GPIO.HIGH)
                time.sleep(1)
            else:
                GPIO.output(led_pin, GPIO.LOW)
                time.sleep(1)

def record(r_led, g_led, event_manager, processing_complete, processed_frames):
    print('recording')
    g_led.Update(100, force=True)
    r_led.Update(1, force=True)
    processing_complete.wait()
    processing_complete.clear()
    event_manager.update(recording)

    errors = 0
    while True: # wait here until all frames have been processed and sent to server
        try: 
            # get the processed frames from queue
            n_frame, ball_candidates = processed_frames.get_nowait()
            # print(f"rec: {n_frame}")
            message = sf.MyMessage(c.TYPE_BALLS, (n_frame, ball_candidates))
            if not sf.send_message(client.client_socket, message, c.CLIENT):
                errors += 1
                print(f"error: {errors}")
        
        # if processing complete and no more data to send to server
        except queue.Empty:
            if processing_complete.is_set() and (processed_frames.qsize() == 0):
                event_manager.clear()
                break
    # if there were no transmission errors send True, else send False
    if errors == 0:
        message = sf.MyMessage(c.TYPE_DONE, True)
    else:
        message = sf.MyMessage(c.TYPE_DONE, False)
    sf.send_message(client.client_socket, message, c.CLIENT)
    return

def stream(r_led, g_led, event_manager, processing_complete, processed_frames):
    print('recording stream')
    message_list = []
    g_led.Update(1, force=True)
    r_led.Update(1, force=True)
    processing_complete.wait()
    processing_complete.clear()
    event_manager.update(record_stream)
    errors = 0

    while True:
        try:
            # get the frames from queue
            n_frame, frame_buf = processed_frames.get_nowait()
            
            if event_manager.record_stream.is_set():
                print(f"calib: {n_frame}")
                # y_data is a numpy array hxw with 8 bit greyscale brightness values
                y_data = np.frombuffer(frame_buf, dtype=np.float32, count=w*h).reshape((h,w)).astype(np.uint8)
                message = sf.MyMessage(c.TYPE_IMG, (n_frame, y_data))
                if not sf.send_message(client.client_socket, message, c.CLIENT):
                    errors += 1
                    print(f"error: {errors}")

                message_list.extend(client.read_all_server_messages())
                for message in message_list:
                    if message['data'].type == c.TYPE_DONE:
                        event_manager.clear()

        except queue.Empty:
            if not event_manager.record_stream.is_set() and processed_frames.qsize() == 0:  # if the recording has finished
                print('stream done')
                processing_complete.set()
                break

    # if there were no transmission errors send True, else send False
    if errors == 0:
        message = sf.MyMessage(c.TYPE_DONE, True)
    else:
        message = sf.MyMessage(c.TYPE_DONE, False)

    sf.send_message(client.client_socket, message, c.CLIENT)



def shutdown_prog(r_led, g_led, event_manager):
    r_led.Kill()
    g_led.Kill()
    event_manager.update()
    event_manager.shutdown.set()
    print('shutdown (main)')
    return


if __name__ == "__main__":
    message_list = []

    ## -- setup client connection to server -- ##

    client.connect_to_server(name=client_name) 

    ## -- initialise multithreading objs -- ##
    unprocessed_frames = mp.Queue()
    processed_frames = mp.Queue()
    shutdown = mp.Event()
    picam_ready = mp.Event()
    processing_complete = mp.Event()

    event_change = mp.Event()
    recording = mp.Event()
    record_stream = mp.Event()
    capture_img = mp.Event()

    event_manager = EventManager(event_change, recording, record_stream, capture_img, shutdown)

    t_record = mp.Value('i',c.REC_T)
    n_calib = mp.Value('i', int(c.FRAMERATE*c.CALIB_IMG_DELAY))

    r_led = LED(c.R_LED_PIN, 0, shutdown)
    g_led = LED(c.G_LED_PIN, 0, shutdown)

    processing_complete.set() # inintially there is no processing being done

    ## -- initialise Picam process for recording
    Picam = mp.Process(target=StartPicam, args=(unprocessed_frames, processed_frames, picam_ready, processing_complete, t_record, n_calib, event_manager))
    Picam.start()
    picam_ready.wait()

    while True and not shutdown.is_set():
        try:
            message_list = []
            r_led.Update(0)
            g_led.Update(1)
            ## -- read server messages -- ##
            message_list.extend(client.read_all_server_messages())

            for message in message_list:
                try:
                    # record message
                    if message['data'].type == c.TYPE_REC:
                        # change the recording time if the new recording time is valid
                        if isinstance(message['data'].message, int):
                            t_record.value = message['data'].message
                        else:
                            t_record.value = c.REC_T
                        record(r_led, g_led, event_manager, processing_complete, processed_frames)
                        break
                    elif message['data'].type == c.TYPE_STREAM:
                        n_calib.value = int(c.FRAMERATE*message['data'].message)
                        stream(r_led, g_led, event_manager, processing_complete, processed_frames)
                        break

                    elif message['data'].type == c.TYPE_SHUTDOWN:
                        processing_complete.clear()
                        shutdown_prog(r_led, g_led, event_manager)
                        processing_complete.wait()
                        Picam.terminate()
                        Picam.join()
                        print('shutdown complete')
                        break

                except Exception as e:
                    print(e)
                    print('unrecognised message type')
                    continue

        except sf.CommError as e:
            traceback.print_exc(file=sys.stdout)
            processing_complete.clear()
            shutdown_prog(r_led, g_led, event_manager)
            processing_complete.wait()
            Picam.terminate()
            Picam.join()
            print('shutdown complete')
            break