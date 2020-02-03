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
T_RECORD_DEF = 1    # default recording time

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
                
                ## -- do some processing -- ##
                y_data_out = y_data
                ##--------------------------##
                processed_frames.put((n_frame, y_data_out))

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
            self.n_frame = 0    # reset frame number when recording is finished

    def flush(self):
        for i, processor in enumerate(self.processors):
            self.proc_complete[i].wait() # wait until process has finished
            processor.terminate()
            processor.join()
        print('shutdown complete')

def StartPicam(unprocessed_frames, processed_frames, recording, shutdown, picam_ready, processing_complete, t_record):
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
                    for proc in proc_complete:  # reset all processing complete events
                        proc.clear()
                    camera.wait_recording(t_record.value) # record for an amount of time
                finally:
                    recording.clear()   # clear the record flag to stop processing
                    for proc in proc_complete:  # wait for all processing to be complete
                        proc.wait()
                    processing_complete.set()   # set processing complete event
            elif shutdown.is_set():
                print('shutdown')
                break
        camera.stop_recording()
        print('not recording') 
    return

if __name__ == "__main__":
    ## -- setup client connection to server -- ##
    client.connect_to_server(name=c.LEFT_CLIENT) 

    ## -- initialise multithreading objs -- ##
    unprocessed_frames = mp.Queue()
    processed_frames = mp.Queue()
    recording = mp.Event()
    shutdown = mp.Event()
    picam_ready = mp.Event()
    processing_complete = mp.Event()
    t_record = mp.Value('i',T_RECORD_DEF)

    state = c.STATE_IDLE    # sets the default state

    ## -- initialise Picam process for recording
    Picam = mp.Process(target=StartPicam, args=(unprocessed_frames, processed_frames, recording, shutdown, picam_ready, processing_complete, t_record))
    Picam.start()

    while True:
        if state == c.STATE_IDLE:
            ## -- read server messages -- ##
            message_list = client.read_all_server_messages()
            for message in message_list:
                try:
                    print(message['data'])
                    print(message['data'].type)
                    print(message['data'].message)
                    # record message
                    if message['data'].type == c.TYPE_REC:
                        state = c.STATE_RECORDING
                        # change the recording time if the new recording time is valid
                        if isinstance(message['data'].message, int):
                            t_record.value = message['data'].message
                        else:
                            t_record.value = T_RECORD_DEF
                        break   # go and do the recording, ignore other messages 
                except:
                    print('unrecognised message type')
                    continue
        elif state == c.STATE_RECORDING:
            picam_ready.wait()  # waits for the picam to initialise
            processing_complete.clear()
            recording.set()

            errors = 0
            while True: # wait here until all frames have been processed and sent to server
                try: 
                    # get the processed frames from queue
                    frame_n, y_data = processed_frames.get_nowait()
                    m = frame_n
                    message = sf.MyMessage(c.TYPE_BALLS, m)
                    print('sending')
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
            print(message.message)

            state = c.STATE_IDLE    # reset the state to IDLE and wait for next instruction
            continue