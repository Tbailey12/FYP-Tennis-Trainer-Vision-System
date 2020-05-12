import socket
import select
import time
import os
import sys
import queue
import cv2
import multiprocessing as mp
import numpy as np
from inspect import signature
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import consts as c
import funcs as func
import socket_funcs as sf
import stereo_calibration as s_cal
import triangulation as tr

right_client_bypass = True

class Server(object):
    def __init__(self):
        self.root_p = os.getcwd()
        self.socket = self.create_server_socket()
        self.stereo_calib = self.load_stereo_calib(c.ACTIVE_STEREO_F)
        self.sockets_list = [self.socket]    # list of sockets, init with server socket
        self.clients = {}                    # dict of key:socket, value:client_name
        self.client_names = []

        self.cmd_parser = {
            c.TYPE_RECORD: self.record,
            c.TYPE_STREAM: self.stream,
            c.TYPE_SHUTDOWN: self.shutdown,
            c.TYPE_HELP: self.help_func
        }

    def create_server_socket(self):
        '''
        Creates a socket for communication with clients
        '''
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # create IPV4 socket for server
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # allows us to reconnect to same port

        server_socket.bind((c.SERVER_IP, c.PORT))
        server_socket.listen()
        return server_socket

    def add_new_client(self):
        '''
        Adds a new client to the clients dict and adds the corresponding socket to sockets list
        Return: True if new client
                False if no new client
        '''
        client_socket, client_address = self.socket.accept()
        try:
            client = sf.receive_message(client_socket, c.SERVER)
            if client is not None:
                self.sockets_list.append(client_socket)
                self.clients[client_socket] = client
                print(f"Accepted connection from client: {client['data']} on {client_address[0]}:{client_address[1]}")
                self.client_names.append(client['data'])
                return True
            else:
                return False

        except sf.CommError as e:
            print(e.message)
            return False

    def send_to_client(self, client_name, message):
        '''
        Send message to the client specified by client_name

        Return: True if sent, False if error
        '''
        try:
            for client_socket in self.sockets_list:
                if client_socket != self.socket:  # if the socket is not the server socket
                    # send left message
                    if self.clients[client_socket]['data'] == client_name:
                        c.print_debug(f"Sending message to {client_name} client: Time: {message}")
                        sf.send_message(client_socket, message, c.SERVER)
            return True

        except sf.CommError as e:
            raise sf.CommError(e)
            return False

    def read_client_messages(self, read_all=False):
        '''
        Read messages from all connected clients
        read_all=False (default) - One message at a time
        read_all=True - Read all messages at once

        Return list of messages [{header: message_header, data: message_data}]
        '''
        message_list = []
        read_sockets, _, exception_sockets = select.select(self.sockets_list, [], self.sockets_list, 0)

        while read_sockets:
            for notified_socket in read_sockets:
                # new client connected
                if notified_socket == self.socket:  # client has connected, so accept and handle connection
                    self.add_new_client()
                # existing client connected
                else:
                    message = sf.receive_message(notified_socket, c.SERVER)

                    if message is None:
                        print(f"Closed connection from {self.clients[notified_socket]['data']}")
                        self.sockets_list.remove(notified_socket)
                        del self.clients[notified_socket]
                        continue

                    client = self.clients[notified_socket]
                    c.print_debug(f"Received message from {client['data']}: {message['data']}")
                    message_list.append({"client": client['data'], "data": message['data']})

            # if there is an exception, remove the socket from the list
            for notified_socket in exception_sockets:
                self.sockets_list.remove(notified_socket)
                del self.clients[notified_socket]

            if read_all:
                read_sockets, _, exception_sockets = select.select(self.sockets_list, [], self.sockets_list, 0)
                continue
            else:
                break

        return message_list

    def load_stereo_calib(self, filename):
        stereo_calib = s_cal.StereoCal()
        try:
            os.chdir(func.make_path(self.root_p, c.DATA_DIR, c.STEREO_CALIB_DIR))
            stereo_calib.load_params(filename)
            os.chdir(self.root_p)
            return stereo_calib

        except ValueError as e:
            print(e)
            return None

    def initialise(self):
        '''
        Waits for both left and right client to be connected to the server

        Return: True if initialised, False if not
        '''
        while True:
            self.read_client_messages()

            if len(self.clients) >= 1:
                left_connected, right_connected = False, False

                for client_socket in self.clients:
                    left_connected = True if self.clients[client_socket]['data'] == c.LEFT_CLIENT else left_connected
                    right_connected = True if self.clients[client_socket]['data'] == c.RIGHT_CLIENT else right_connected

                if left_connected and right_connected:
                    print('both clients connected')
                    return True

                elif left_connected and right_client_bypass:
                    print('left client connected')
                    return True

            time.sleep(0.01)
        return False

    def initialise_picamera(self):
        '''
        Sends message to both clients to start the picameras
        Waits for confirmation of camera startup

        Return: True if cameras started
        False: Cameras not started
        '''
        print('Initialising cameras')
        message = sf.MyMessage(c.TYPE_START_CAM, None)
        setup_complete = {}

        for client in self.client_names:
            self.send_to_client(client, message)
            setup_complete[client] = False

        while True:
            message_list = self.read_client_messages()
            for message in message_list:
                check_task_trigger(setup_complete, c.TYPE_START_CAM, message)

            if check_task_complete(setup_complete):
                print('Cameras initialised')
                return True

            time.sleep(0.001)

    def record(self, record_t):
        '''
        Sends a message to both clients to start a recording for record_t seconds
        '''
        record_t = convert_string_to_float(record_t)
        if record_t is None: return False

        print(f"Recording for {record_t} seconds")
        message = sf.MyMessage(c.TYPE_RECORD, (record_t,))

        recording_complete = {}
        for client in self.client_names:
            self.send_to_client(client, message)
            recording_complete[client] = False

        ball_candidate_dict = dict.fromkeys(self.client_names)
        for key in ball_candidate_dict.keys():
            ball_candidate_dict[key] = {}

        while True:
            message_list = self.read_client_messages()
            for message in message_list:
                if not check_task_trigger(recording_complete, c.TYPE_DONE, message):
                    check_ball_cand(message, ball_candidate_dict, self.stereo_calib)
                    print(message['data'].message)

            if check_task_complete(recording_complete):
                print('Recording complete')
                points_3d = tr.triangulate_points(ball_candidate_dict, self.stereo_calib)
                for point in points_3d:
                    print(point)
                return True

            time.sleep(0.001)

    def stream(self, stream_t, save=False, show=False):
        '''
        Sends a message to both clients to start a stream for stream_t seconds
        '''
        stream_t = convert_string_to_float(stream_t)
        if stream_t is None: return False

        print(f"Streaming for {stream_t} seconds")
        message = sf.MyMessage(c.TYPE_STREAM, (stream_t,))

        stream_complete = {}
        for client in self.client_names:
            self.send_to_client(client, message)
            stream_complete[client] = False

        img_queue = mp.Queue()
        img_play = mp.Event()

        image_process = mp.Process(target=image_viewer, args=(img_queue, img_play, c.STREAM_DELTA_T))
        image_process.start()
        img_play.set()

        img_dict = dict.fromkeys(self.client_names)
        for key in img_dict.keys():
            img_dict[key] = {}

        cur_frame = 0

        while True:
            message_list = self.read_client_messages()
            for message in message_list:
                if not check_task_trigger(stream_complete, c.TYPE_DONE, message) \
                                        and check_img(message):
                    if save:
                        save_img(message)

                    if show:
                        add_to_img_dict(img_dict, message)
                        img = None
                        img = combine_img(img_dict, cur_frame)

                        if img is not None: 
                            show_img(img, img_queue)
                            cur_frame += c.STREAM_IMG_DELTA

            if check_task_complete(stream_complete) and img_queue.qsize() == 0:
                print('Stream complete')
                cv2.destroyAllWindows()
                image_process.terminate()
                return True

            time.sleep(0.01)

    def shutdown(self):
        '''
        Sends a shutdown command to both clients then shuts down the server

        Return: 
        '''
        print('Shutting down...')
        message = sf.MyMessage(c.TYPE_SHUTDOWN, None)
        for client_name in self.client_names:
            self.send_to_client(client_name, message)

        sys.exit()

    def print_func_info(self, func):
        print(f"Command: {func}{str(signature(self.cmd_parser[func]))}")
        if self.cmd_parser[func].__doc__ is not None:
            print(f"    {self.cmd_parser[func].__doc__}\n")

    def help_func(self, func=None):
        '''
        Prints a list of all server commands
        '''
        if func is None:
            for cmd in self.cmd_parser.keys():
                self.print_func_info(cmd)

        elif func in self.cmd_parser.keys():
            self.print_func_info(func)

        else:
            print(f"'{func}' is not a valid server command, type help for list of commands")

    def cmd_func(self, cmd):
        '''
        Attempts to call the function cmd from self.cmd_parser with any additional args if they exist
        '''
        split = cmd.strip().split(" ")
        func = self.cmd_parser.get(split[0], None)

        if func is None:
            print("Invalid server command, type help for list of commands")
            return

        params = signature(func).parameters
        num_args = len(params)
        num_non_default_args = count_non_default_args(params)

        if len(split[1:]) < num_non_default_args:
            print(f"Too few args. {func.__name__} takes {num_non_default_args} argument{'s'*(num_non_default_args!=1)}")
            print(f"format: {func.__name__}{str(signature(func))}")

        elif len(split[1:]) > num_args:
            print(f"Too many args. {func.__name__} takes {num_args} argument{'s'*(num_args!=1)}")
            print(f"format: {func.__name__}{str(signature(func))}")

        else:
            return func(*split[1:])

def count_non_default_args(args):
    count = 0
    for arg in args.values():
        if arg.default is arg.empty:
            count+=1

    return count

def combine_img(img_dict, cur_frame):
    try:
        keys = list(img_dict.keys())
        if len(keys) > 1:            
            img_comb = cv2.hconcat([img_dict[c.LEFT_CLIENT][cur_frame], img_dict[c.RIGHT_CLIENT][cur_frame]])
            return img_comb

        else:
            img = img_dict[keys[0]][cur_frame]
            return img
    
    except KeyError as e:
        return None

    return None


def add_to_img_dict(img_dict, message):
    n_frame, img = message['data'].message
    img_dict[message['client']][n_frame] = img
    return True

def image_viewer(img_queue, play, deltaT):
    '''
    Displays all frames from img_queue with deltaT (ms) spacing between each frame
    Only displays while play is set
    '''
    while True:
        play.wait()
        try:
            img = img_queue.get_nowait()
            cv2.imshow('img', np.uint8(img))
            cv2.waitKey(30)

        except queue.Empty:
            time.sleep(0.03)

def convert_string_to_float(my_string):
    try:
        return float(my_string)

    except ValueError as e:
        print(e)
        return None

def check_ball_cand(message, ball_candidate_dict, stereo_calib):
    '''
    Checks if received message is ball candidates, if yes, it rectifies the candidate points and adds
    the candidates to the ball_candidate_dict with inner key = client and outer key = frame num
    '''
    if message['data'].type == c.TYPE_BALLS:
        rectify_points(message['data'].message[1], *stereo_calib.get_params(message['client']))
        ball_candidate_dict[message['client']][message['data'].message[0]] = message['data'].message[1]

def check_img(message):
    if message['data'].type == c.TYPE_IMG:
        return True
    else:
        return False

def show_img(img, img_queue):
    img_queue.put(img)
    return True

def save_img(message):
    n_frame, img = message['data'].message
    cv2.imwrite(f"{message['client']}_{n_frame:04d}.png", img)
    return True

def check_task_trigger(task_status_dict, trigger_type, message):
    if  message['data'].type == trigger_type and \
        message['data'].message == True:

        if message['client'] in task_status_dict:
            task_status_dict[message['client']] = True
            return True

    return False

def check_task_complete(task_status_dict):
    all_done = False
    for client in task_status_dict:
        if task_status_dict[client] is False:
            all_done = False
            break
        else:
            all_done = True
            continue

    if all_done:
        return True
    else:
        return False

def rectify_points(frame, camera_matrix, dist_coeffs, R_matrix, P_matrix, *args):
    for candidate in frame:
        if candidate is not []:
            candidate[c.X_COORD:c.Y_COORD+1] = cv2.undistortPoints(candidate[c.X_COORD:c.Y_COORD+1], camera_matrix, dist_coeffs, R=R_matrix, P=P_matrix)

if __name__ == "__main__":
    print("Server started")
    server = Server()
    server.initialise()
    server.initialise_picamera()
    while True:
        try:
            message_list = server.read_client_messages(read_all=True)

            for message in message_list:
                print(message['data'].message)

            cmd = input("cmd: ")
            server.cmd_func(cmd)

        except sf.CommError as e:
            print("Connection to client closed unexpectedly, shutting down...")
            quit()

    # def search_for_chessboards(chessboards_found, chessboards, left_frame, right_frame):
#     left_chessboard = s_cal.find_chessboards(left_frame)
#     if left_chessboard:
#         right_chessboard = s_cal.find_chessboards(right_frame)
#     if left_chessboard and right_chessboard:
#         chessboards.put((left_chessboard[0], right_chessboard[0]))
#         chessboards_found.value += 1
#         print(f'chessboards found {chessboards_found.value}')
#     return

# def server_help():
#     helpstring = '''
#     Server cmd list:
#     help - display list of server commands
#     calibrate - calibrates the stereo cameras using 10 images
#     '''
#     return helpstring

# def rectify_points(frame, camera_matrix, dist_coeffs, R_matrix, P_matrix):
#     for candidate in frame:
#         if candidate is not []:
#             candidate[c.X_COORD:c.Y_COORD+1] = cv2.undistortPoints(candidate[c.X_COORD:c.Y_COORD+1], camera_matrix, dist_coeffs, R=R_matrix, P=P_matrix)

# def triangulate_points(stereo_calib, left_candidates, right_candidates):
#     # rotation and translation matrices to account for camera baseline, height and angle
#     Rx = np.array(  [[1,        0,                      0,                      0],
#                     [0,         np.cos(c.CAM_ANGLE),    -np.sin(c.CAM_ANGLE),   0],
#                     [0,         np.sin(c.CAM_ANGLE),    np.cos(c.CAM_ANGLE),    0],
#                     [0,         0,                      0,                      1]], dtype=np.float32)

#     Tz = np.array([ [1, 0, 0, -(c.CAM_BASELINE/2)],
#                     [0, 1, 0, 0],
#                     [0, 0, 1, c.CAM_HEIGHT],
#                     [0, 0, 0, 1]], dtype=np.float32)

#     Rx = Rx.dot(Tz)

#     candidates_3D = len(left_candidates)*[[]]

#     for f, frame in enumerate(left_candidates):
#         print(f"frame: {f}")
#         if left_candidates[f] is None or right_candidates[f] is None:
#             candidates_3D[f] = []
#             continue
#         else:
#             candidates = []
#             for i_l, c_l in enumerate(left_candidates[f]):
#                 max_sim = 0
#                 print(f"left: {i_l}")
#                 for j_r, c_r in enumerate(right_candidates[f]):
#                     print(f"right: {j_r}")
#                     if abs(c_l[c.Y_COORD]-c_r[c.Y_COORD]) < c.DISP_Y:

#                         # calculate ratio between left and right candidate
#                         width_r = c_l[c.WIDTH]/c_r[c.WIDTH]
#                         size_r = c_l[c.SIZE]/c_r[c.SIZE]
#                         height_r = c_l[c.HEIGHT]/c_r[c.HEIGHT]

#                         # ensure all ratios are > 1
#                         if width_r<1: width_r=1/width_r
#                         if size_r<1: size_r=1/size_r
#                         if height_r<1: height_r=1/height_r

#                         calc_b = ((size_r)**2 + (width_r)**2 + (height_r)**2)

#                         if calc_b != 0:
#                             sim = 3/calc_b
#                         else:
#                             sim = np.Inf
                
#                         # find the object in the right image with the highest similarity
#                         if sim>max_sim:
#                             r_match = j_r
#                             max_sim = sim

#                 if max_sim > c.SIM_THRESH:
#                     points4d = cv2.triangulatePoints(stereo_calib.P1, stereo_calib.P2, left_candidates[f][i_l][c.X_COORD:c.Y_COORD+1], right_candidates[f][r_match][c.X_COORD:c.Y_COORD+1]).flatten()
#                     points3d = [i/points4d[3] for i in points4d[:3]]
#                     points3d_shift = [points3d[0], points3d[2], -points3d[1], 1]
#                     points3d_shift = Rx.dot(points3d_shift)
#                     candidates.append(points3d_shift)

#             candidates_3D[f] = candidates
#     return candidates_3D

# def plot_points(points_3d):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_xlabel('x (m)')
#     ax.set_ylabel('y (m)')
#     ax.set_zlabel('z (m)')
#     ax.set_xlim(-11E-1/2, 11E-1/2)
#     ax.set_ylim(0, 24E-1)
#     ax.set_zlim(0, 2E-1)

#     for f, candidates in enumerate(points_3d):
#         for candidate in candidates:
#             ax.scatter(xs=candidate[0],ys=candidate[1],zs=candidate[2])

#     plt.show()

# def record(stereo_calib = None, record_time = c.REC_T):
#     if stereo_calib is None:
#         print('stereo cameras must first be calibrated')
#         return False

#     message_list = []
#     pos = 0
#     left_done = False
#     right_done = False

#     right_frames = 0
#     left_frames = 0

#     rec_obj = sf.MyMessage(c.TYPE_REC, record_time)
#     send_to_client(c.LEFT_CLIENT, rec_obj)
#     send_to_client(c.RIGHT_CLIENT, rec_obj)

#     left_candidates = record_time*100*[None]
#     right_candidates = record_time*100*[None]

#     while True:
#         message_list.extend(read_all_client_messages())

#         while pos < len(message_list):
#             if message_list[pos]['data'].type == c.TYPE_BALLS:

#                 if message_list[pos]['client'] == c.LEFT_CLIENT:
#                     f = int(message_list[pos]['data'].message[0])
#                     rectify_points(message_list[pos]['data'].message[1], stereo_calib.cameraMatrix1, stereo_calib.distCoeffs1, stereo_calib.R1, stereo_calib.P1)
#                     left_candidates[f] = message_list[pos]['data'].message[1]
#                     left_frames+=1

#                 elif message_list[pos]['client'] == c.RIGHT_CLIENT:
#                     f = int(message_list[pos]['data'].message[0])
#                     rectify_points(message_list[pos]['data'].message[1], stereo_calib.cameraMatrix2, stereo_calib.distCoeffs2, stereo_calib.R2, stereo_calib.P2)
#                     right_candidates[f] = message_list[pos]['data'].message[1]
#                     right_frames+=1
                
#             elif message_list[pos]['data'].type == c.TYPE_DONE:
#                 if message_list[pos]['client'] == c.LEFT_CLIENT:
#                     left_done = True
#                 elif message_list[pos]['client'] == c.RIGHT_CLIENT:
#                     right_done = True
#             else:
#                 print(f"unknown message format for recording: {message_list[pos]['data'].type}")
#             pos+=1

#         if left_done and right_done:
#             print('recording finished')
#            # -- Make both lists the same length -- #
#             min_length = min(left_frames, right_frames)
#             left_candidates = left_candidates[:min_length]
#             right_candidates = right_candidates[:min_length]

#             print(f"frames captured: {min_length}")

#             return left_candidates, right_candidates

# def stream(run_time = c.CALIB_T, calibrate = False, display = False, timeout = False):
#     message_list = []
#     left_stream_imgs = []
#     right_stream_imgs = []
#     left_chessboards = []
#     right_chessboards = []
#     chessboards = mp.Queue()
#     chessboard_searchers = []
#     pos = 0
#     left_done = False
#     right_done = False
#     disp_n_frame = 0
#     cal_n_frame = 0
#     img_size = None
#     done = False
#     stopframe = int(run_time*c.FRAMERATE)
#     chessboards_found = mp.Value('i',0)

#     rec_obj = sf.MyMessage(c.TYPE_STREAM, c.CALIB_IMG_DELAY)
#     send_to_client(c.LEFT_CLIENT, rec_obj)
#     send_to_client(c.RIGHT_CLIENT, rec_obj)

#     if calibrate:
#         # load camera intrinsic calibration data
#         left_cal, right_cal = s_cal.load_calibs()
        
#     while True:
#         message_list.extend(read_all_client_messages())
#         if len(message_list) > 0:
#             while pos < len(message_list):
#                 # when the clients send an image during calibration
#                 if (message_list[pos]['data'].type == c.TYPE_IMG) and not done:
#                     n_frame = message_list[pos]['data'].message[0]
#                     print(n_frame)
#                     y_data = message_list[pos]['data'].message[1]
#                     if img_size is None:
#                         (h,w) = y_data.shape[:2]
#                         img_size = (w,h)
#                     # add the img to the corresponding calibration img list
#                     if message_list[pos]['client'] == c.LEFT_CLIENT:
#                         left_stream_imgs.append((n_frame, y_data))
#                     elif message_list[pos]['client'] == c.RIGHT_CLIENT:                    
#                         right_stream_imgs.append((n_frame, y_data))
#                     # cv2.imwrite(f"{message_list[pos]['client']}{n_frame}.png",y_data)

#                     if display:
#                         if (len(left_stream_imgs) > disp_n_frame) and (len(right_stream_imgs) > disp_n_frame): 
                            
#                             ########## FOR TESTING ##############
#                             os.chdir(c.IMG_P)
#                             cv2.imwrite(f"l_{(disp_n_frame):04d}.png",left_stream_imgs[disp_n_frame][1])
#                             cv2.imwrite(f"r_{(disp_n_frame):04d}.png",right_stream_imgs[disp_n_frame][1])
#                             os.chdir(c.ROOT_P)
#                             ########## FOR TESTING ##############

#                             disp_frame = cv2.hconcat([left_stream_imgs[disp_n_frame][1],right_stream_imgs[disp_n_frame][1]])
#                             cv2.imshow(f"stream", disp_frame)
#                             print(disp_n_frame)
#                             cv2.waitKey(100)
#                             if left_stream_imgs[disp_n_frame][0] >=stopframe and timeout:
#                                 done_obj = sf.MyMessage(c.TYPE_DONE, 1)
#                                 send_to_client(c.LEFT_CLIENT, done_obj)
#                                 send_to_client(c.RIGHT_CLIENT, done_obj)
#                                 done = True
#                                 cv2.destroyAllWindows()
                            
#                             disp_n_frame += 1

#                     # look for chessboards
#                     if calibrate:
#                         if (len(left_stream_imgs) > cal_n_frame) and (len(right_stream_imgs) > cal_n_frame):
#                             chessboard_search = mp.Process(target = search_for_chessboards, args=(chessboards_found, chessboards, [left_stream_imgs[cal_n_frame]], [right_stream_imgs[cal_n_frame]]))
#                             chessboard_search.start()
#                             chessboard_searchers.append(chessboard_search)
#                             cal_n_frame += 1

#                     if chessboards_found.value >= c.MIN_PATTERNS:
#                         done_obj = sf.MyMessage(c.TYPE_DONE, 1)
#                         send_to_client(c.LEFT_CLIENT, done_obj)
#                         send_to_client(c.RIGHT_CLIENT, done_obj)
#                         if display: 
#                             done = True
#                             cv2.destroyAllWindows()

#                 # when both clients send the done message, they are finished collecting frames
#                 elif (message_list[pos]['data'].type == c.TYPE_DONE):
#                     if message_list[pos]['client'] == c.LEFT_CLIENT:
#                         left_done = True
#                     elif message_list[pos]['client'] == c.RIGHT_CLIENT:
#                         right_done = True
#                     if left_done and right_done:
#                         if calibrate and chessboards_found.value >= c.MIN_PATTERNS:
#                             # for searcher in chessboard_searchers:
#                             #     searcher.join()
#                             while True:
#                                 try:
#                                     left_chessboard, right_chessboard = chessboards.get_nowait()
#                                     left_chessboards.append(left_chessboard)
#                                     right_chessboards.append(right_chessboard)
#                                 except queue.Empty:
#                                     if chessboards.qsize() == 0:
#                                         break

#                             # # check all chessboards are valid in both images
#                             s_cal.validate_chessboards(left_chessboards, right_chessboards)
#                             # calibrated stereo cameras
#                             RMS, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = s_cal.calibrate_stereo(
#                                 left_chessboards, right_chessboards, left_cal, right_cal, img_size)
    
#                             # obtain stereo rectification projection matrices
#                             R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1,
#                                                         cameraMatrix2, distCoeffs2, img_size, R, T)

#                             # save all calibration params to object
#                             stereo_calib =  s_cal.StereoCal(RMS, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F,
#                                                         R1, R2, P1, P2, Q, validPixROI1, validPixROI2)

#                             s_cal.save_stereo_calib(stereo_calib)
                        
#                             print(f'calibration complete, rms: {RMS}')
#                             return stereo_calib
#                         else:
#                             return None
#                 pos += 1


    # time.sleep(3)
    # print('initialised')
    # stereo_calib = s_cal.load_stereo_calib()
    
    # while True:
    #     time.sleep(1/1000)

    #     cmd = input("Enter server cmd: ")
    #     if cmd == "help":
    #         print(server_help())
    #     if cmd == "calibrate":
    #         while True:
    #             cmd = input("Load existing calibration? (y/n)")
    #             if cmd == "y":
    #                 stereo_calib = s_cal.load_stereo_calib()
    #                 break
    #             elif cmd == "n":
    #                 cal_result = stream(run_time=60, calibrate=True, display=True, timeout=True)
    #                 if cal_result:
    #                     stereo_calib = cal_result
    #                 break
    #             else:
    #                 print(f"{cmd} is not a valid cmd, please try again")

    #     elif cmd == "stream":
    #         while True:
    #             try: 
    #                 cmd = int(input("stream time: "))
    #             except ValueError:
    #                 print("Invalid type, please enter an integer for time")
    #                 continue
    #             if cmd <= c.STREAM_MAX:
    #                 stream(run_time=cmd, calibrate=False, display=True, timeout=True)
    #                 break
    #             else:
    #                 print(f"Please enter a time less than {c.STREAM_MAX}")

    #     elif cmd == "record":
    #         if stereo_calib.rms is None:
    #             print("calibration must be conducted before recording")
    #         else:
    #             cmd = input("Recording time: ")
    #             try:
    #                 cmd = int(cmd)
    #                 if cmd < c.REC_T_MAX:
    #                     left_candidates, right_candidates = record(stereo_calib = stereo_calib, record_time = cmd)
    #                     points_3d = triangulate_points(stereo_calib, left_candidates, right_candidates)
    #                     np.save('candidates_3D.npy', points_3d)
    #                     print('analyse_tracklets')
    #                     params = stt.analyse_tracklets(points_3d)

    #                     if params is None:
    #                         print('No tracklets found')
    #                     else:
    #                         t = np.linspace(0,cmd,1000)

    #                         x_params = params[0]
    #                         y_params = params[1]
    #                         z_params = params[2]
    #                         x_points = params[3]
    #                         y_points = params[4]
    #                         z_points = params[5]

    #                         x_est = stt.curve_func(t,*x_params)
    #                         y_est = stt.curve_func(t,*y_params)
    #                         z_est = stt.curve_func(t,*z_params)

    #                         xd1_est = stt.d1_curve_func(t,*x_params)
    #                         yd1_est = stt.d1_curve_func(t,*y_params)
    #                         zd1_est = stt.d1_curve_func(t,*z_params)

    #                         z_min = 100
    #                         for i, z in enumerate(z_est):
    #                             if z<=0:
    #                                 bounce_pos = i
    #                                 break
    #                             else:
    #                                 if z<z_min:
    #                                     z_min = z
    #                             bounce_pos = z_min

    #                         x_vel = xd1_est[bounce_pos]
    #                         y_vel = yd1_est[bounce_pos]
    #                         z_vel = zd1_est[bounce_pos]

    #                         print(x_vel,y_vel,z_vel)
    #                         print(f"velocity: {np.sqrt(x_vel**2+y_vel**2+z_vel**2):2.2f} m/s")
    #                         print(f"bounce_loc: {x_est[bounce_pos]:0.2f}, {y_est[bounce_pos]:0.2f},{z_est[bounce_pos]:0.2f}")

    #                         ## -- Plot points -- ##
    #                         import matplotlib.pyplot as plt
    #                         from mpl_toolkits.mplot3d import Axes3D
    #                         fig = plt.figure()
    #                         ax = fig.add_subplot(111, projection='3d')
    #                         ax.set_xlabel('x (m)')
    #                         ax.set_ylabel('y (m)')
    #                         ax.set_zlabel('z (m)')
    #                         ax.set_xlim(-11E-1/2, 11E-1/2)
    #                         ax.set_ylim(0, 24E-1)
    #                         ax.set_zlim(0, 3E-1)

    #                         ax.scatter(xs=x_points,ys=y_points,zs=z_points,c=np.arange(len(x_points)), cmap='winter')

    #                         z_est[bounce_pos:] = None

    #                         ax.plot3D(x_est,y_est,z_est)
    #                         plt.show()

    #                 else:
    #                     raise ValueError('invalid time entered')
    #             except ValueError:
    #                 print(f"Invalid recording time, please enter time in s less than {c.REC_T_MAX}")

    #     elif cmd == "shutdown":
    #         shutdown()
    #     else:
    #         print(f"{cmd} is not a valid command, enter 'help' to see command list")
    #     continue
