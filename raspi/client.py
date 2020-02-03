import socket
import select
import errno
import sys
import time
from datetime import datetime

import consts as c
import socket_funcs as sf

debug = c.DEBUG

# name = c.LEFT_CLIENT

def print_debug(my_print):
    if debug:
        print(my_print) 
 

def connect_to_server(name):
    # attempt to connect to the server, if connection refused, wait 1s then try again
    while True:
        try:
            client_socket.connect((c.IP, c.PORT))  # connect to server
        except socket.error as e:
            print("Connection could not be established to server, trying again...")
            time.sleep(5)
            continue
        except socket.timeout as e:
            print("Connection could not be established to server, trying again...")
            time.sleep(5)
            continue
        except Exception as e:
            print('Error', str(e))
            sys.exit()
        print(f"Connection Established to server on {c.IP}:{c.PORT}")
        client_socket.setblocking(True)
        sf.send_message(client_socket, name, c.CLIENT)  # send name to server
        time.sleep(1)
        break


def read_all_server_messages():
    message_list = []
    # ---    receive all messages from the server    ---- #
    message_recv = sf.receive_message(client_socket, c.CLIENT)  # receive messages from the server
    while message_recv is not None:
        message_list.append(message_recv)
        message_recv = sf.receive_message(client_socket, c.CLIENT)  # receive messages from the server

    return message_list


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # create IPV4 socket for
client_socket.settimeout(c.SOCKET_TIMEOUT)

if __name__ == "__main__":
    connect_to_server()
    counter = 0
    while True:
        time.sleep(1 / 100)

        # ----   send messages to the server     ---- #
        message = f"[{counter}] Time:{datetime.now()}"
        print_debug(f"Sending message to Server: {message}")
        if not sf.send_message(client_socket, message, c.CLIENT):
            sys.exit()  # there was a problem sending the message

        message_list = read_all_server_messages()
        for message in message_list:
            try:
                if message['data'].type == c.TYPE_REC:
                    print('record')
                    print(message['data'].message)
                elif message['data'].type == c.TYPE_CAP:
                    print('capture')
                    print(message['data'].message)
            except:
                print('unrecognised message format')

        counter += 1