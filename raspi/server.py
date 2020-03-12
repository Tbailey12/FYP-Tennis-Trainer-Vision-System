'''
socket tutorial (SentDex) https://www.youtube.com/watch?v=ytu2yV3Gn1I
'''

import socket
import select
import time
import sys
from datetime import datetime

import cv2 as cv

import consts as c
import socket_funcs as sf
# import camera_calibration as cal
import stereo_calibration as s_cal

import multiprocessing as mp
import queue

import numpy as np
import os

def send_to_client(client_name, message):
    for client_socket in sockets_list:
        if client_socket != server_socket:  # if the socket is not the server socket
            # send left message
            if clients[client_socket]['data'] == client_name:
                c.print_debug(f"Sending message to {client_name} client: Time: {message}")
                sf.send_message(client_socket, message, c.SERVER)

def read_all_client_messages():
    message_list = []
    # syntax for select.select()
    # (sockets we read, sockets we write, sockets that error)
    read_sockets, _, exception_sockets = select.select(sockets_list, [], sockets_list, 0)
    # read all messages from clients
    while read_sockets:
        # if any of the read_sockets have new data
        for notified_socket in read_sockets:
            # new client connected
            if notified_socket == server_socket:  # client has connected, so accept and handle connection
                client_socket, client_address = server_socket.accept()

                client = sf.receive_message(client_socket, c.SERVER)
                if client is None:  # client disconnected while sending
                    continue
                sockets_list.append(client_socket)  # append new socket to list of client sockets
                clients[client_socket] = client
                print(
                    f"Accepted new connection from {client_address[0]}:{client_address[1]}, client:{client['data']}")
            # existing client connected
            else:
                message = sf.receive_message(notified_socket, c.SERVER)

                if message is None:
                    print(f"Closed connection from {clients[notified_socket]['data']}")
                    sockets_list.remove(notified_socket)
                    del clients[notified_socket]
                    continue

                client = clients[notified_socket]
                c.print_debug(f"Received message from {client['data']}: {message['data']}")
                message_list.append({"client": client['data'], "data": message['data']})

        # if there is an exception, remove the socket from the list
        for notified_socket in exception_sockets:
            sockets_list.remove(notified_socket)
            del clients[notified_socket]

        # if there are more messages to be read, read them
        # read_sockets, _, exception_sockets = select.select(sockets_list, [], sockets_list, 0)
        return message_list
    return []

def search_for_chessboards(chessboards_found, chessboards, left_frame, right_frame):
    left_chessboard = s_cal.find_chessboards(left_frame)
    if left_chessboard:
        right_chessboard = s_cal.find_chessboards(right_frame)
    if left_chessboard and right_chessboard:
        chessboards.put((left_chessboard[0], right_chessboard[0]))
        chessboards_found.value += 1
        print(f'chessboards found {chessboards_found.value}')
    return

def server_help():
    helpstring = '''
    Server cmd list:
    help - display list of server commands
    calibrate - calibrates the stereo cameras using 10 images
    '''
    return helpstring

def initialise():
    message_list = []
    left_connected = False
    right_connected = False

    while True:
        message_list.extend(read_all_client_messages())
        for client_socket in sockets_list:
            if client_socket != server_socket:  # if the socket is not the server socket
                if clients[client_socket]['data'] == c.LEFT_CLIENT:
                    left_connected = True
                elif clients[client_socket]['data'] == c.RIGHT_CLIENT:
                    right_connected = True

        if left_connected and right_connected:
            print('both clients connected')
            return True


def record(stereo_calib = None, record_time = c.REC_T):
    # if stereo_calib is None:
    #     print('stereo cameras must first be calibrated')

    message_list = []
    pos = 0
    left_done = False
    right_done = False

    right_frame = 0
    left_frame = 0

    ##################### TESTING #####################
    left_balls = []
    right_balls = []
    # right_done = True
    ####################################################

    rec_obj = sf.MyMessage(c.TYPE_REC, record_time)
    send_to_client(c.LEFT_CLIENT, rec_obj)
    send_to_client(c.RIGHT_CLIENT, rec_obj)
    ####################################################
    ####################################################

    while True:
        message_list.extend(read_all_client_messages())
        while pos < len(message_list):
            if message_list[pos]['data'].type == c.TYPE_BALLS:
                # print(message_list[pos]['data'].message[0])
                if message_list[pos]['client'] == c.LEFT_CLIENT:
                    left_frame = message_list[pos]['data'].message[0]
                    left_balls.append(message_list[pos]['data'].message)
                elif message_list[pos]['client'] == c.RIGHT_CLIENT:
                    right_frame = message_list[pos]['data'].message[0]
                    right_balls.append(message_list[pos]['data'].message)

            elif message_list[pos]['data'].type == c.TYPE_DONE:
                if message_list[pos]['client'] == c.LEFT_CLIENT:
                    left_done = True
                elif message_list[pos]['client'] == c.RIGHT_CLIENT:
                    right_done = True
                if left_done and right_done:
                    print('recording finished')
                    print(f"left frames: {left_frame}")
                    print(f"right frames: {right_frame}")
                    ##################### TESTING #####################
                    np.save('left_ball_candidates.npy', left_balls)
                    np.save('right_ball_candidates.npy', right_balls)
                    ###################################################
                    return True
            else:
                print(f"unknown message format for recording: {message_list[pos]['data'].type}")
            pos+=1

def stream(run_time = c.CALIB_T, calibrate = False, display = False, timeout = False):
    message_list = []
    left_stream_imgs = []
    right_stream_imgs = []
    left_chessboards = []
    right_chessboards = []
    chessboards = mp.Queue()
    chessboard_searchers = []
    pos = 0
    left_done = False
    right_done = False
    disp_n_frame = 0
    cal_n_frame = 0
    img_size = None
    done = False
    stopframe = int(run_time*c.FRAMERATE)
    chessboards_found = mp.Value('i',0)

    rec_obj = sf.MyMessage(c.TYPE_STREAM, c.CALIB_IMG_DELAY)
    send_to_client(c.LEFT_CLIENT, rec_obj)
    send_to_client(c.RIGHT_CLIENT, rec_obj)

    if calibrate:
        # load camera intrinsic calibration data
        left_cal = s_cal.CamCal()
        right_cal = s_cal.CamCal()
        left_cal.load_params(c.LEFT_CALIB_F)
        right_cal.load_params(c.RIGHT_CALIB_F)
    while True:
        message_list.extend(read_all_client_messages())
        if len(message_list) > 0:
            while pos < len(message_list):
                # when the clients send an image during calibration
                if (message_list[pos]['data'].type == c.TYPE_IMG) and not done:
                    n_frame = message_list[pos]['data'].message[0]
                    print(n_frame)
                    y_data = message_list[pos]['data'].message[1]
                    if img_size is None:
                        (h,w) = y_data.shape[:2]
                        img_size = (w,h)
                    # add the img to the corresponding calibration img list
                    if message_list[pos]['client'] == c.LEFT_CLIENT:
                        left_stream_imgs.append((n_frame, y_data))
                    elif message_list[pos]['client'] == c.RIGHT_CLIENT:                    
                        right_stream_imgs.append((n_frame, y_data))
                    # cv.imwrite(f"{message_list[pos]['client']}{n_frame}.png",y_data)

                    if display:
                        if (len(left_stream_imgs) > disp_n_frame) and (len(right_stream_imgs) > disp_n_frame): 
                            
                            ########## FOR TESTING ##############
                            os.chdir(c.IMG_P)
                            cv.imwrite(f"l_{(disp_n_frame+100):04d}.png",left_stream_imgs[disp_n_frame][1])
                            # cv.imwrite(f"r_{time.time_ns():04d}.png",right_stream_imgs[disp_n_frame][1])
                            os.chdir(c.ROOT_P)
                            ########## FOR TESTING ##############

                            disp_frame = cv.hconcat([left_stream_imgs[disp_n_frame][1],right_stream_imgs[disp_n_frame][1]])
                            cv.imshow(f"stream", disp_frame)
                            print(disp_n_frame)
                            cv.waitKey(100)
                            if left_stream_imgs[disp_n_frame][0] >=stopframe and timeout:
                                done_obj = sf.MyMessage(c.TYPE_DONE, 1)
                                send_to_client(c.LEFT_CLIENT, done_obj)
                                send_to_client(c.RIGHT_CLIENT, done_obj)
                                done = True
                                cv.destroyAllWindows()
                            
                            disp_n_frame += 1

                    # look for chessboards
                    if calibrate:
                        if (len(left_stream_imgs) > cal_n_frame) and (len(right_stream_imgs) > cal_n_frame):
                            chessboard_search = mp.Process(target = search_for_chessboards, args=(chessboards_found, chessboards, [left_stream_imgs[cal_n_frame]], [right_stream_imgs[cal_n_frame]]))
                            chessboard_search.start()
                            chessboard_searchers.append(chessboard_search)
                            cal_n_frame += 1

                    if chessboards_found.value >= c.MIN_PATTERNS:
                        done_obj = sf.MyMessage(c.TYPE_DONE, 1)
                        send_to_client(c.LEFT_CLIENT, done_obj)
                        send_to_client(c.RIGHT_CLIENT, done_obj)
                        if display: 
                            done = True
                            cv.destroyAllWindows()

                # when both clients send the done message, they are finished collecting frames
                elif (message_list[pos]['data'].type == c.TYPE_DONE):
                    if message_list[pos]['client'] == c.LEFT_CLIENT:
                        left_done = True
                    elif message_list[pos]['client'] == c.RIGHT_CLIENT:
                        right_done = True
                    if left_done and right_done:
                        if calibrate and chessboards_found.value >= c.MIN_PATTERNS:
                            # for searcher in chessboard_searchers:
                            #     searcher.join()
                            while True:
                                try:
                                    left_chessboard, right_chessboard = chessboards.get_nowait()
                                    left_chessboards.append(left_chessboard)
                                    right_chessboards.append(right_chessboard)
                                except queue.Empty:
                                    if chessboards.qsize() == 0:
                                        break

                            # # check all chessboards are valid in both images
                            s_cal.validate_chessboards(left_chessboards, right_chessboards)
                            # calibrated stereo cameras
                            RMS, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = s_cal.calibrate_stereo(
                                left_chessboards, right_chessboards, left_cal, right_cal, img_size)
    
                            # obtain stereo rectification projection matrices
                            R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(cameraMatrix1, distCoeffs1,
                                                        cameraMatrix2, distCoeffs2, img_size, R, T)

                            # save all calibration params to object
                            stereo_calib =  s_cal.StereoCal(RMS, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F,
                                                        R1, R2, P1, P2, Q, validPixROI1, validPixROI2)

                            stereo_calib.save_params(c.STEREO_CALIB_F)
                            print(f'calibration complete, rms: {RMS}')
                            return stereo_calib
                        else:
                            return None
                pos += 1

def shutdown():
    print('shutting down')
    shut_obj = sf.MyMessage(c.TYPE_SHUTDOWN, None)
    send_to_client(c.LEFT_CLIENT, shut_obj)
    send_to_client(c.RIGHT_CLIENT, shut_obj)
    while True:
        time.sleep(2)
        sys.exit()

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # create IPV4 socket for server
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # allows us to reconnect to same port

server_socket.bind((c.IP, c.PORT))
server_socket.listen()

sockets_list = [server_socket]  # list of sockets, init with server socket
clients = {}  # list of clients

j = 0

if __name__ == "__main__":
    print("Server started")
    initialise()
    time.sleep(3)
    print('initialised')
    stereo_calib = s_cal.StereoCal()
    ########## FOR TESTING ##############
    stereo_calib.load_params(c.STEREO_CALIB_F)
    ########## FOR TESTING ##############
    
    while True:
        time.sleep(1/1000)

        cmd = input("Enter server cmd: ")
        if cmd == "help":
            print(server_help())
        if cmd == "calibrate":
            while True:
                cmd = input("Load existing calibration? (y/n)")
                if cmd == "y":
                    stereo_calib.load_params(c.STEREO_CALIB_F)
                    break
                elif cmd == "n":
                    cal_result = stream(run_time=60, calibrate=True, display=True, timeout=True)
                    if cal_result:
                        stereo_calib = cal_result
                    break
                else:
                    print(f"{cmd} is not a valid cmd, please try again")

        elif cmd == "stream":
            while True:
                try: 
                    cmd = int(input("stream time: "))
                except ValueError:
                    print("Invalid type, please enter an integer for time")
                    continue
                if cmd <= c.STREAM_MAX:
                    stream(run_time=cmd, calibrate=False, display=True, timeout=True)
                    break
                else:
                    print(f"Please enter a time less than {c.STREAM_MAX}")

        elif cmd == "record":
            if stereo_calib.rms is None:
                print("calibration must be conducted before recording")
            else:
                record()
        elif cmd == "shutdown":
            shutdown()
        else:
            print(f"{cmd} is not a valid command, enter 'help' to see command list")
        continue