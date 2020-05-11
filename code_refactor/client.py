import socket
import select
import errno
import sys
import time
import queue
import traceback
from inspect import signature
import multiprocessing as mp

import consts as c
import socket_funcs as sf
import camera_utils
import l_r_consts

class Client(object):
    def __init__(self, client_name):
        print(f"Client: {client_name} started")
        self.name = client_name
        self.socket = self.create_client_socket()
        self.connect_to_server()
        self.camera_manager = None
        self.camera_process = None

        self.cmd_parser = {
            c.TYPE_START_CAM: self.initialise_picamera,
            c.TYPE_RECORD: self.record,
            c.TYPE_STREAM: self.stream,
            c.TYPE_SHUTDOWN: self.shutdown
        }
    def connect_to_server(self):
        '''
        Attempt to connect to the server, if connection refused, wait 1s then try again

        Return: True once connected
        '''
        while True:
            try:
                print(f"Connecting to server on {c.SERVER_IP}:{c.PORT}...")
                self.socket.connect((c.SERVER_IP, c.PORT))  # connect to server
                print(f"Connection Established to server on {c.SERVER_IP}:{c.PORT}")
                sf.send_message(self.socket, self.name, c.CLIENT)  # send name to server
                return True

            except (socket.timeout, TimeoutError) as e:
                print("Connection could not be established to server, trying again...")
                continue

            except socket.error as e:
                if e.errno != errno.EALREADY:
                    raise e

            except Exception as e:
                print('Error', str(e))
                # raise sf.CommError("Communication Error") from e

    def create_client_socket(self):
        '''
        Creates the client socket for communication with the server

        Return: client_socket
        '''
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # create IPV4 socket
        client_socket.settimeout(c.SOCKET_TIMEOUT)
        client_socket.setblocking(True)
        return client_socket

    def read_server_messages(self,read_all=False):
        '''
        Read all messages from the socket connected to the server

        Return: List of messages [{header: message_header, data: message_data}]
        '''
        message_list = []
        # ---    receive all messages from the server    ---- #
        try:
            read_sockets, _, exception_sockets = select.select([self.socket], [], [self.socket], 0)
            while read_sockets:
                for notified_socket in read_sockets:
                    message = sf.receive_message(notified_socket, c.CLIENT)
                    if message is None:
                        print(f"Closed connection from {c.SERVER}")
                        continue
                    message_list.append(message)

                if read_all:
                    read_sockets, _, exception_sockets = select.select([self.socket], [], [self.socket], 0)
                    continue
                else:
                    break
            return message_list

        except sf.CommError as e:
            raise sf.CommError("Communication Error") from e

    def send_to_server(self, message_type, message_data):
        '''
        Send message to the server given the message type and data

        Return True if sent successfully
        '''
        message = sf.MyMessage(message_type, message_data)
        sf.send_message(self.socket, message, c.CLIENT)
        return True

    def cmd_func(self, cmd, args=None):
        '''
        Attempts to call the function cmd from self.cmd_parser with any additional args if they exist
        '''
        func = self.cmd_parser.get(cmd, None)

        if args == None:
            return func()

        elif len(signature(func).parameters) == len(args):
                return func(*args)

        else: return    

    def initialise_picamera(self):
        '''
        Initialise the picamera with default values

        Return: True if successful
                False if not successful
        '''
        if self.camera_manager == None:
            print('Initialising camera')
            self.camera_manager = camera_utils.CameraManager()
            self.camera_process = mp.Process(target=self.camera_manager.start_camera, args=())
            self.camera_process.start()

            self.camera_manager.event_manager.picam_ready.wait()
            self.camera_manager.event_manager.processing_complete.set()
            self.send_to_server(c.TYPE_START_CAM, True)
            print("Camera initialised")
            return True
        
        else:
            print('Error: Camera already initialised')
            return False


    def record(self, record_t):
        '''
        Records for record_t using the existing camera object and sends all ball candidate data to the server
        Sends TYPE_DONE when no more messages to send
        '''
        print(f"Recording for {record_t} seconds")
        self.camera_manager.record(record_t)
        while True:
            try:
                ball_data = self.camera_manager.frame_queues.processed_frames.get_nowait()
                self.send_to_server(c.TYPE_BALLS, ball_data)

            except queue.Empty:
                if  self.camera_manager.event_manager.processing_complete.is_set() and \
                            not self.camera_manager.event_manager.recording.is_set():
                    self.send_to_server(c.TYPE_DONE, True)
                    print('Recording complete')
                    break
            time.sleep(0.001)

    def stream(self, stream_t):
        '''
        Streams for stream_t using the existing camera object and sends all raw frame data to the server
        Sends TYPE_DONE when no more messages to send        
        '''
        print(f"Streaming for {stream_t} seconds")
        self.camera_manager.stream(stream_t)
        while True:
            try:
                stream_data = self.camera_manager.frame_queues.processed_frames.get_nowait()
                self.send_to_server(c.TYPE_IMG, stream_data)

            except queue.Empty:
                if self.camera_manager.event_manager.processing_complete.is_set() and \
                        self.camera_manager.frame_queues.processed_frames.qsize() == 0:
                    self.send_to_server(c.TYPE_DONE, True)
                    print('Stream complete')
                    break

            time.sleep(0.001)

    def shutdown(self):
        '''
        Attempts to gracefully close all processes that are currently running then closes the script
        '''
        print('Shutting down...')
        self.camera_manager.event_manager.shutdown.set()
        self.camera_manager.shutdown.wait(1)
        self.camera_process.kill()
        print("Shutdown (client)")
        sys.exit()

if __name__ == "__main__":
    client_name = l_r_consts.CLIENT_NAME
    if len(sys.argv)>1:
        client_name = sys.argv[1]
    client = Client(client_name)

    while True:
        try:
            message_list = client.read_server_messages()
            for message in message_list:
                client.cmd_func(message['data'].type, message['data'].message)

            time.sleep(0.001)

        except (sf.CommError, KeyboardInterrupt) as e:
            print('Server disconnected, closing client')
            client.shutdown()
            quit()

    # counter = 0
    # while True:
    #     time.sleep(1 / 100)

    #     # ----   send messages to the server     ---- #
    #     message = f"[{counter}] Time:{datetime.now()}"
    #     c.print_debug(f"Sending message to Server: {message}")
    #     if not sf.send_message(client_socket, message, c.CLIENT):
    #         sys.exit()  # there was a problem sending the message

    #     message_list = read_all_server_messages()
    #     for message in message_list:
    #         try:
    #             if message['data'].type == c.TYPE_REC:
    #                 print('record')
    #                 print(message['data'].message)
    #             elif message['data'].type == c.TYPE_CAP:
    #                 print('capture')
    #                 print(message['data'].message)
    #         except:
    #             print('unrecognised message format')

    #     counter += 1