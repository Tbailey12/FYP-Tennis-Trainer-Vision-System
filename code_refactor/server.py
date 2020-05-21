import socket
import select
import time
import os
import sys
import queue
import cv2
import pickle
import multiprocessing as mp
import numpy as np
from inspect import signature
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import consts as c
import funcs as func
import socket_funcs as sf
import stereo_calibration as s_cal
import triangulation as tr
import trajectory_analysis as ta

right_client_bypass = False
root_p = os.getcwd()

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

            time.sleep(0.001)
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
        
        if record_t is not None and isinstance(record_t, float):
            if record_t > 0 and record_t <= c.REC_T_MAX:
                record_t = record_t
            else:
                print(f"ERROR: {record_t}s is too long for recording")
                record_t = c.REC_T_MAX
        else:
            return False

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

            if check_task_complete(recording_complete):
                print('Recording complete')
                with open("ball_dict.pkl", "wb") as file:
                    pickle.dump(ball_candidate_dict, file)

                points_3d = tr.triangulate_points(ball_candidate_dict, self.stereo_calib)

                plot_points_3d(points_3d)
                np.save("points_3d.npy", points_3d)

                trajectory = analyse_trajectory(points_3d)
                if trajectory is not None:
                    plot_trajectory(trajectory, save=False)

                return True

            time.sleep(0.001)

    def stream(self, stream_t, show=False, save=False):
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

            time.sleep(0.001)

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

def plot_points_3d(points_3d):
    plt.close('all')
    fig = plt.figure(figsize=(15*1.25,4*1.25))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_xlim(c.XMIN/2, c.XMAX/2)
    ax.set_ylim(0, c.YMAX)
    ax.set_zlim(0, c.ZMAX)
    ax.view_init(elev=20,azim=-20)

    for frame in points_3d:
        for point in frame:
            if len(point) > 0:
                ax.scatter(xs=point[0],ys=point[1],zs=point[2])

    plt.show()


def plot_trajectory(trajectory, save=False):
    fig = plt.figure('trajectory', figsize=(15*1.25,4*1.25))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_xlim(c.XMIN/2, c.XMAX/2)
    ax.set_ylim(0, c.YMAX)
    ax.set_zlim(0, c.ZMAX)
    ax.view_init(elev=20,azim=-20)

    for tok in trajectory['tracklet'].tokens:
        ax.scatter(xs=tok.coords[c.X_3D],ys=tok.coords[c.Y_3D],zs=tok.coords[c.Z_3D],c='blue',alpha=0.2)

    ax.plot3D(trajectory['curve']['x'], trajectory['curve']['y'], trajectory['curve']['z'], c='red')
    plt.show()

def find_start_end_pos(curve):
    start, end = None, None
    for i, point in enumerate(curve['y']):
        if point >= 0 and curve['z'][i] > 0:
            start = i
            break

    for i, point in enumerate(curve['z']):
        if point < 0 and i>start:
           end = i
           break
           
    return start, end

def trim_curve(curve, key_points):
    start = key_points[0]
    end = key_points[1]

    if start is None or end is None:
        return False

    else:
        for key in curve:
            curve[key] = curve[key][start:end]
        return True

def print_bounce_stats(curve, curve_d1):
    position = {'t': curve['t'][-1], 'x': curve['x'][-1], 'y': curve['y'][-1], 'z': curve['z'][-1]}
    velocity = ta.calc_dist([curve_d1['x'][-1], curve_d1['y'][-1], curve_d1['z'][-1]])

    print(f"Bounce Pos: t: {position['t']:0.2f}s, x: {position['x']:0.2f}m, y: {position['y']:0.2f}m, z: {position['z']:0.2f}m")
    print(f"Bounce velocity: {velocity:0.2f}m/s")

def analyse_trajectory(points_3d):
    best_tracklet = ta.get_tracklets(points_3d)

    if best_tracklet is None:
        print("No valid trajectories found")
        return None

    best_tracklet = ta.split_tracklet(best_tracklet)
    curve_params = fit_curve(best_tracklet)

    if curve_params is None:
        print("Curve could not be fitted")
        return None

    curve = est_points(curve_params, ta.curve_func, t=(0, c.SHOT_T_MAX, c.FIT_POINTS))
    curve_d1 = est_points(curve_params, ta.d1_curve_func, t=(0, c.SHOT_T_MAX, c.FIT_POINTS))

    if curve is None or curve_d1 is None:
        return None

    key_points = find_start_end_pos(curve)
    trim_curve(curve, key_points)
    trim_curve(curve_d1, key_points)
    print_bounce_stats(curve, curve_d1)

    trajectory = {'tracklet': best_tracklet, 'curve': curve, 'curve_d1': curve_d1}
    return trajectory

def est_points(curve_params, curve_func, t=(0, c.SHOT_T_MAX, c.FIT_POINTS)):
    t_points = np.linspace(t[0],t[1],t[2])

    try:
        x_est = curve_func(t_points, *curve_params['x'])
        y_est = curve_func(t_points, *curve_params['y'])
        z_est = curve_func(t_points, *curve_params['z'])

        curve = {'t': t_points, 'x': x_est, 'y': y_est, 'z': z_est}
        return curve

    except Exception as e:
        print(f"Exception: {e}")
        return None

def fit_curve(tracklet):
    if tracklet is None:
        return None

    x_points = []
    y_points = []
    z_points = []

    for i, tok in enumerate(tracklet.tokens):
        x_points.append(tok.coords[c.X_3D])
        y_points.append(tok.coords[c.Y_3D])
        z_points.append(tok.coords[c.Z_3D])

    t = np.linspace(tracklet.start_frame/c.FRAMERATE, \
                (tracklet.start_frame+tracklet.length)/c.FRAMERATE, \
                tracklet.length)

    x_params, covmatrix = curve_fit(ta.curve_func, t, x_points, method='lm')
    y_params, covmatrix = curve_fit(ta.curve_func, t, y_points, method='lm')
    z_params, covmatrix = curve_fit(ta.curve_func, t, z_points, method='lm')

    curve_params = {'x':x_params, 'y':y_params, 'z':z_params}
    
    return curve_params

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
            cv2.waitKey(1)

        except queue.Empty:
            time.sleep(0.001)
            pass

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
    os.chdir(func.make_path(root_p, c.IMG_DIR, c.STREAM_DIR))
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
    func.clean_dir(func.make_path(root_p, c.IMG_DIR, c.STREAM_DIR))
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