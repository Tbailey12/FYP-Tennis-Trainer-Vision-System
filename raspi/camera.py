import io
import time
# import threading
import picamera
import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
import multiprocessing as mp
import queue
import cv2
import os

import client
import consts as c
import socket_funcs as sf

## -- camera settings -- ##
# w,h = (1280,720)
w,h = (640,480)
resolution = w,h
framerate = 90
n_processors = 4    # number of processors to use for CV

def ImageProcessor(unprocessed_frames, processed_frames, recording, proc_complete, calibration):
    processing = False

    while True:
        if not processing:
            recording.wait()    # wait for a recording to start
            if not calibration.is_set():
                processing = True
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
        camera.annotate_frame_num = True
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

    state = c.STATE_IDLE    # sets the default state

    ##############################################################################
    # state = c.STATE_CALIBRATION    # sets the default state
    ##############################################################################

    ## -- initialise Picam process for recording
    Picam = mp.Process(target=StartPicam, args=(unprocessed_frames, processed_frames, recording, shutdown, picam_ready, processing_complete, t_record, calibration))
    Picam.start()

    while True:

        if state == c.STATE_IDLE:
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
                    elif message['data'].type == c.TYPE_SHUTDOWN:
                        state = c.STATE_SHUTDOWN
                        break
                except:
                    print('unrecognised message type')
                    continue

                    
        elif state == c.STATE_RECORDING:
            print('recording')
            picam_ready.wait()  # waits for the picam to initialise
            processing_complete.clear()
            recording.set()

            errors = 0
            while True: # wait here until all frames have been processed and sent to server
                try: 
                    # get the processed frames from queue
                    frame_n, y_data = processed_frames.get_nowait()
                    m = [(frame_n,i,i) for i in range(1000)]
                    message = sf.MyMessage(c.TYPE_BALLS, m)
                    if not sf.send_message(client.client_socket, message, c.CLIENT):
                        errors += 1
                        print(f"error: {errors}")
                
                # if processing complete and no more data to send to server
                except queue.Empty:
                    if processing_complete.is_set():
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
            time.sleep(2)   # wait for person to get ready with calib board
            processing_complete.clear()
            calibration.set()
            recording.set()
            t_record.value = c.CALIB_T

            while True:
                try:
                    # get the frames from queue
                    n_frame, frame_buf = processed_frames.get_nowait()
                    # y_data is a numpy array hxw with 8 bit greyscale brightness values
                    y_data = np.frombuffer(frame_buf, dtype=np.uint8, count=w*h).reshape((h,w))
                    cv2.imwrite(f"{n_frame}.png", y_data)
                except queue.Empty:
                    if not recording.is_set():  # if the recording has finished
                        break
            t_record.value = c.REC_T
            calibration.clear()

            state = c.STATE_SHUTDOWN

        elif state == c.STATE_SHUTDOWN:
            print('shutdown (main)')
            shutdown.set()
            break