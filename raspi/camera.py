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
framerate = 90
n_processors = 4    # number of processors to use for CV

def ImageProcessor(unprocessed_frames, processed_frames, recording, proc_complete, calibration):
    processing = False
    proc_complete.set()

    while True:
        if not processing:
            recording.wait()    # wait for a recording to start
            if not calibration.is_set():
                processing = True
                proc_complete.clear()
            else:
                proc_complete.set()
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
                if not recording.is_set():  # if the recording has finished
                    processing = False      # reset the processing flag
                    proc_complete.set()     # set the proc_complete event

class FrameController(object):
    def __init__(self, unprocessed_frames, processed_frames, recording, proc_complete, calibration):
        self.unprocessed_frames = unprocessed_frames
        self.processed_frames = processed_frames
        self.recording = recording
        self.calibration = calibration
        self.proc_complete = proc_complete
        self.number_of_processors = n_processors
        self.processors = []
        self.n_frame = 0

        ## -- create processors for processing the individual video frames
        for i in range(n_processors):
            proc_event = mp.Event()
            processor = mp.Process(target=ImageProcessor, args=(self.unprocessed_frames,self.processed_frames,self.recording,proc_event,self.calibration))
            self.proc_complete.append(proc_event)
            self.processors.append(processor)
            processor.start()

    def write(self, buf):
        if self.recording.is_set():
            if self.calibration.is_set():
                if self.n_frame%framerate == 0:
                    self.processed_frames.put((self.n_frame, buf))
            else:
                self.unprocessed_frames.put((self.n_frame, buf))    # add the new frame to the queue
            self.n_frame += 1   # increment the frame number
        else:
            self.n_frame = 0    # reset frame number when recording is finished

    def flush(self):
        for i, processor in enumerate(self.processors):
            self.proc_complete[i].wait() # wait until process has finished
            processor.terminate()
            processor.join()
        print('shutdown complete')


def StartPicam(unprocessed_frames, processed_frames, recording, shutdown, picam_ready, processing_complete, t_record, calibration):
    proc_complete = []     # contains processing events, true if complete
    with picamera.PiCamera() as camera:
        camera.framerate = framerate
        camera.resolution = resolution
        camera.vflip = True # camera is upside down so flip the image
        ## -- adds the frame number to the image for testing
        # camera.annotate_frame_num = True
        camera.annotate_text_size = 160
        camera.annotate_foreground = picamera.Color('green')
        ## ----------------------------------------------- ##
        time.sleep(2)   # give the camera a couple of seconds to initialise

        output = FrameController(unprocessed_frames, processed_frames, recording, proc_complete, calibration)
        camera.start_recording(output, format='yuv')

        picam_ready.set()   # the picam is ready to record

        # Picam will stay in this while loop waiting for either shutdown or recording flags to be set
        while True:
            if recording.wait(1):    # wait for recording flag to be set in main()
                processing_complete.clear()
                try:
                    for proc in proc_complete:  # reset all processing complete events
                        proc.clear()
                    camera.wait_recording(t_record.value) # record for an amount of time
                finally:
                    recording.clear()   # clear the record flag to stop processing
                    for proc in proc_complete:  # wait for all processing to be complete
                        proc.wait()
                    processing_complete.set()   # set processing complete event
            elif shutdown.is_set():
                print('shutdown (picam)')
                break
        camera.stop_recording()
    return


class LED(object):
    def __init__(self, led_pin, led_freq, shutdown):
        self.led_pin = led_pin
        self.led_freq = led_freq
        self.shutdown = shutdown
        self.process = None

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(led_pin, GPIO.OUT)

    def Update(self, led_freq):
        if self.led_freq == led_freq:
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


if __name__ == "__main__":
    message_list = []

    ## -- setup client connection to server -- ##
    client.connect_to_server(name=c.LEFT_CLIENT) 

    ## -- initialise multithreading objs -- ##
    unprocessed_frames = mp.Queue()
    processed_frames = mp.Queue()
    recording = mp.Event()
    calibration = mp.Event()
    shutdown = mp.Event()
    picam_ready = mp.Event()
    processing_complete = mp.Event()
    t_record = mp.Value('i',c.REC_T)

    r_led = LED(c.R_LED_PIN, 0, shutdown)
    g_led = LED(c.G_LED_PIN, 0, shutdown)

    state = c.STATE_IDLE    # sets the default state
    processing_complete.set() # inintially there is no processing being done

    ## -- initialise Picam process for recording
    Picam = mp.Process(target=StartPicam, args=(unprocessed_frames, processed_frames, recording, shutdown, picam_ready, processing_complete, t_record, calibration))
    Picam.start()

    while True:
        try:
            if state == c.STATE_IDLE:
                r_led.Update(0)
                g_led.Update(1)
                ## -- read server messages -- ##
                message_list.extend(client.read_all_server_messages())

                for message in message_list:
                    try:
                        # record message
                        if message['data'].type == c.TYPE_REC:
                            state = c.STATE_RECORDING
                            # change the recording time if the new recording time is valid
                            if isinstance(message['data'].message, int):
                                t_record.value = message['data'].message
                            else:
                                t_record.value = c.REC_T
                            message_list = []
                            break   # go and do the recording, ignore other messages
                        elif message['data'].type == c.TYPE_CALIB:
                            message_list = []
                            state = c.STATE_CALIBRATION
                            break
                        elif message['data'].type == c.TYPE_SHUTDOWN:
                            state = c.STATE_SHUTDOWN
                            break
                    except:
                        print('unrecognised message type')
                        continue

                        
            elif state == c.STATE_RECORDING:
                print('recording')
                g_led.Update(100)
                r_led.Update(1)
                picam_ready.wait()  # waits for the picam to initialise
                processing_complete.wait()
                print('recording set')
                recording.set()

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

                state = c.STATE_IDLE    # reset the state to IDLE and wait for next instruction
                continue


            elif state == c.STATE_CALIBRATION:
                print('calibrating')
                picam_ready.wait()  # waits for the picam to initialise
                g_led.Update(1)
                time.sleep(2)   # wait for person to get ready with calib board
                r_led.Update(1)
                processing_complete.wait()
                calibration.set()
                t_record.value = c.CALIB_T
                recording.set()

                errors = 0

                while True:
                    try:
                        # get the frames from queue
                        n_frame, frame_buf = processed_frames.get_nowait()
                        print(f"calib: {n_frame}")
                        # y_data is a numpy array hxw with 8 bit greyscale brightness values
                        y_data = np.frombuffer(frame_buf, dtype=np.uint8, count=w*h).reshape((h,w))
                        message = sf.MyMessage(c.TYPE_IMG, (n_frame, y_data))
                        if not sf.send_message(client.client_socket, message, c.CLIENT):
                            errors += 1
                            print(f"error: {errors}")

                    except queue.Empty:
                        if not recording.is_set():  # if the recording has finished
                            break

                # if there were no transmission errors send True, else send False
                if errors == 0:
                    message = sf.MyMessage(c.TYPE_DONE, True)
                else:
                    message = sf.MyMessage(c.TYPE_DONE, False)
                sf.send_message(client.client_socket, message, c.CLIENT)
                
                t_record.value = c.REC_T
                calibration.clear()

                state = c.STATE_IDLE


        except sf.CommError as e:
            traceback.print_exc(file=sys.stdout)
            state = c.STATE_SHUTDOWN
            continue


        if state == c.STATE_SHUTDOWN:
            r_led.Update(10)
            g_led.Update(0)
            print('shutdown (main)')
            shutdown.set()
            break