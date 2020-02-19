import io
import time
# import threading
import traceback
import picamera
import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
import multiprocessing as mp
import queue
import cv2
import os
import sys

import client
import consts as c
import socket_funcs as sf

import RPi.GPIO as GPIO

## -- camera settings -- ##
# w,h = (1280,720)
w,h = (640,480)
resolution = w,h
framerate = c.FRAMERATE
n_processors = 4    # number of processors to use for CV

client_name = c.LEFT_CLIENT

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

def ImageProcessor(unprocessed_frames, processed_frames, proc_complete, event_manager):
    processing = False
    proc_complete.set()

    while True:
        if not processing:
            event_manager.event_change.wait()    # wait for a recording to start
            if event_manager.recording.is_set():
                proc_complete.clear()
                processing = True
            # if not calibration.is_set():
            #     processing = True
            #     proc_complete.clear()
            # else:
            #     proc_complete.set()
        else:
            try:
                # get the frames from queue
                n_frame, frame_buf = unprocessed_frames.get_nowait()
                # y_data is a numpy array hxw with 8 bit greyscale brightness values
                y_data = np.frombuffer(frame_buf, dtype=np.uint8, count=w*h).reshape((h,w))
                
                ## -- do some processing -- ##
                y_data_out = y_data
                ##--------------------------##
                processed_frames.put((n_frame, y_data_out))

            except queue.Empty:
                if not event_manager.recording.is_set() and unprocessed_frames.qsize() == 0:  # if the recording has finished
                    processing = False      # reset the processing flag
                    proc_complete.set()     # set the proc_complete event

class FrameController(object):
    def __init__(self, unprocessed_frames, processed_frames, proc_complete, n_calib, event_manager):
        self.unprocessed_frames = unprocessed_frames
        self.processed_frames = processed_frames
        self.n_calib = n_calib
        self.event_manager = event_manager
        self.proc_complete = proc_complete
        self.number_of_processors = n_processors
        self.processors = []
        self.n_frame = 0

        ## -- create processors for processing the individual video frames
        for i in range(n_processors):
            proc_event = mp.Event()
            processor = mp.Process(target=ImageProcessor, args=(self.unprocessed_frames,self.processed_frames, proc_event, self.event_manager))
            self.proc_complete.append(proc_event)
            self.processors.append(processor)
            processor.start()

    def write(self, buf):
        if self.event_manager.recording.is_set():
            self.unprocessed_frames.put((self.n_frame, buf))    # add the new frame to the queue
            self.n_frame += 1
        else:
            self.n_frame = 0


        # if self.recording.is_set():
        #     if self.calibration.is_set():
        #         if self.n_frame%n_calib.value == 0:
        #             self.processed_frames.put((self.n_frame, buf))
        #     else:
        #         self.unprocessed_frames.put((self.n_frame, buf))    # add the new frame to the queue
        #     self.n_frame += 1   # increment the frame number
        # else:
        #     self.n_frame = 0    # reset frame number when recording is finished

    def flush(self):
        for i, processor in enumerate(self.processors):
            self.proc_complete[i].wait() # wait until process has finished
            processor.terminate()
            processor.join()
        print('shutdown complete')


def StartPicam(unprocessed_frames, processed_frames, picam_ready, processing_complete, t_record, n_calib, event_manager):
    proc_complete = []     # contains processing events, true if complete
    with picamera.PiCamera() as camera:
        camera.framerate = framerate
        camera.resolution = resolution
        camera.vflip = True # camera is upside down so flip the image
        camera.hflip = True # flip horizontal too
        ## -- adds the frame number to the image for testing
        # camera.annotate_frame_num = True
        camera.annotate_text_size = 160
        camera.annotate_foreground = picamera.Color('green')
        ## ----------------------------------------------- ##
        time.sleep(2)   # give the camera a couple of seconds to initialise

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
                        event_manager.clear()
                        processing_complete.set()   # set processing complete event

                elif event_manager.record_stream.is_set():
                    processing_complete.wait()

                elif event_manager.capture_img.is_set():
                    processing_complete.wait()

                elif event_manager.shutdown.is_set():
                    print('shutdown (picam)')
                    break
        camera.stop_recording()
    return

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
            if self.process.is_alive():
                self.process.kill()
        # start new LED process with the new flash frequency
        self.process = mp.Process(target=self.Run,args=(self.led_pin,self.led_freq,self.shutdown))
        self.process.start()

    def Kill(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.kill()
        GPIO.cleanup()

    def Run(self,led_pin,led_freq,shutdown):
        while True:
            if shutdown.is_set():
                GPIO.cleanup()
                break
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
        return

def record(r_led, g_led, event_manager, processing_complete, processed_frames):
    print('recording')
    g_led.Update(100)
    r_led.Update(1)
    processing_complete.wait()
    processing_complete.clear()
    event_manager.update(recording)

    errors = 0
    while True: # wait here until all frames have been processed and sent to server
        try: 
            # get the processed frames from queue
            n_frame, y_data = processed_frames.get_nowait()
            print(f"rec: {n_frame}")
            m = [(n_frame,i,i) for i in range(1000)]
            message = sf.MyMessage(c.TYPE_BALLS, m)
            if not sf.send_message(client.client_socket, message, c.CLIENT):
                errors += 1
                print(f"error: {errors}")
        
        # if processing complete and no more data to send to server
        except queue.Empty:
            if processing_complete.is_set() and (processed_frames.qsize() == 0):
                break
    # if there were no transmission errors send True, else send False
    if errors == 0:
        message = sf.MyMessage(c.TYPE_DONE, True)
    else:
        message = sf.MyMessage(c.TYPE_DONE, False)
    sf.send_message(client.client_socket, message, c.CLIENT)
    return


def shutdown_prog(r_led, g_led, shutdown):
    r_led.Update(10)
    g_led.Update(0)
    print('shutdown (main)')
    event_manager.update()
    event_manager.shutdown.set()


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
                    # elif message['data'].type == c.TYPE_CALIB:
                    #     cal_message = message['data'].message

                    #     n_calib.value = int(c.FRAMERATE*cal_message.img_delay)
                    #     t_record.value = int(cal_message.img_delay*cal_message.num_img)
                        
                    #     message_list = []
                    #     state = c.STATE_CALIBRATION
                    #     break
                    elif message['data'].type == c.TYPE_SHUTDOWN:
                        shutdown_prog(r_led, g_led, event_manager)
                except:
                    print('unrecognised message type')
                    continue

        except sf.CommError as e:
            traceback.print_exc(file=sys.stdout)
            shutdown_prog(r_led, g_led, event_manager)
                        
            # elif state == c.STATE_RECORDING:
            #     print('recording')
            #     g_led.Update(100)
            #     r_led.Update(1)
            #     picam_ready.wait()  # waits for the picam to initialise
            #     processing_complete.wait()
            #     print('recording set')
            #     processing_complete.clear()
            #     recording.set()

            #     errors = 0
            #     while True: # wait here until all frames have been processed and sent to server
            #         try: 
            #             # get the processed frames from queue
            #             n_frame, y_data = processed_frames.get_nowait()
            #             print(f"rec: {n_frame}")
            #             m = [(n_frame,i,i) for i in range(1000)]
            #             message = sf.MyMessage(c.TYPE_BALLS, m)
            #             if not sf.send_message(client.client_socket, message, c.CLIENT):
            #                 errors += 1
            #                 print(f"error: {errors}")
                    
            #         # if processing complete and no more data to send to server
            #         except queue.Empty:
            #             if processing_complete.is_set() and (processed_frames.qsize() == 0):
            #                 processing_complete.clear()
            #                 break
            #     # if there were no transmission errors send True, else send False
            #     if errors == 0:
            #         message = sf.MyMessage(c.TYPE_DONE, True)
            #     else:
            #         message = sf.MyMessage(c.TYPE_DONE, False)
            #     sf.send_message(client.client_socket, message, c.CLIENT)

            #     state = c.STATE_IDLE    # reset the state to IDLE and wait for next instruction
            #     continue


            # elif state == c.STATE_CALIBRATION:
            #     print('calibrating')
            #     picam_ready.wait()  # waits for the picam to initialise
            #     g_led.Update(1)
            #     time.sleep(2)   # wait for person to get ready with calib board
            #     g_led.Update(1, force=True)
            #     r_led.Update(1)
            #     processing_complete.wait()
            #     calibration.set()
            #     recording.set()

            #     errors = 0

            #     while True:
            #         try:
            #             # get the frames from queue
            #             n_frame, frame_buf = processed_frames.get_nowait()
            #             print(f"calib: {n_frame}")
            #             # y_data is a numpy array hxw with 8 bit greyscale brightness values
            #             y_data = np.frombuffer(frame_buf, dtype=np.uint8, count=w*h).reshape((h,w))
            #             message = sf.MyMessage(c.TYPE_IMG, (n_frame, y_data))
            #             if not sf.send_message(client.client_socket, message, c.CLIENT):
            #                 errors += 1
            #                 print(f"error: {errors}")

            #         except queue.Empty:
            #             if not recording.is_set():  # if the recording has finished
            #                 break

            #     # if there were no transmission errors send True, else send False
            #     if errors == 0:
            #         message = sf.MyMessage(c.TYPE_DONE, True)
            #     else:
            #         message = sf.MyMessage(c.TYPE_DONE, False)
            #     sf.send_message(client.client_socket, message, c.CLIENT)
                
            #     t_record.value = c.REC_T
            #     calibration.clear()

            #     state = c.STATE_IDLE




        # if state == c.STATE_SHUTDOWN:
        #     r_led.Update(10)
        #     g_led.Update(0)
        #     print('shutdown (main)')
        #     shutdown.set()
        #     break