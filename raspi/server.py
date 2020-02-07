'''
socket tutorial (SentDex) https://www.youtube.com/watch?v=ytu2yV3Gn1I
'''

import socket
import select
import time
import sys
from datetime import datetime

import consts as c
import socket_funcs as sf

debug = c.DEBUG


def print_debug(my_print):
    if debug:
        print(my_print)


def send_to_client(client_name, message):
    for client_socket in sockets_list:
        if client_socket != server_socket:  # if the socket is not the server socket
            # send left message
            if clients[client_socket]['data'] == client_name:
                print_debug(f"Sending message to {client_name} client: Time: {message}")
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
                print_debug(f"Received message from {client['data']}: {message['data']}")
                message_list.append({"client": client['data'], "data": message['data']})

        # if there is an exception, remove the socket from the list
        for notified_socket in exception_sockets:
            sockets_list.remove(notified_socket)
            del clients[notified_socket]

        # if there are more messages to be read, read them
        # read_sockets, _, exception_sockets = select.select(sockets_list, [], sockets_list, 0)
        return message_list
    return []
 
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # create IPV4 socket for server
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # allows us to reconnect to same port

server_socket.bind((c.IP, c.PORT))
server_socket.listen()

print("Server started")

sockets_list = [server_socket]  # list of sockets, init with server socket
clients = {}  # list of clients

j = 0

if __name__ == "__main__":
    message_list = []

    state = c.STATE_STOP
    last_len = 0
    while True:
        time.sleep(1/1000)


        if state == c.STATE_IDLE:
            rec_obj = sf.MyMessage(c.TYPE_REC, 1)
            print('starting recording')
            time.sleep(2)
            # state = c.STATE_RECORDING
            # send_to_client(c.LEFT_CLIENT, rec_obj)
            continue
        

        elif state == c.STATE_RECORDING:
            message_list.extend(read_all_client_messages())
            while last_len < len(message_list):
                if type(message_list[last_len]['data'].message) is list:
                    print(message_list[last_len]['data'].message[0])
                else:
                    print(message_list[last_len]['data'].message)
                last_len+=1
            last_len = len(message_list)
            if len(message_list) > 0:
                if(message_list[-1]['data'].type == c.TYPE_DONE):
                    print(message_list[-1]['data'].message)
                    state = c.STATE_SHUTDOWN
                    continue


        elif state == c.STATE_STOP:
            message_list.extend(read_all_client_messages())
            for socket in sockets_list:
                for client_socket in sockets_list:
                    if client_socket != server_socket:  # if the socket is not the server socket
                        if clients[client_socket]['data'] == c.LEFT_CLIENT:
                            state = c.STATE_IDLE
                            del message_list[:]


        elif state == c.STATE_SHUTDOWN:
            if j == 0:
                time.sleep(1)
                state = c.STATE_IDLE
                del message_list[:]
                j+=1
                continue
            print('shutting down')
            shut_obj = sf.MyMessage(c.TYPE_SHUTDOWN, None)
            send_to_client(c.LEFT_CLIENT, shut_obj)
            while True:
                time.sleep(1)
                sys.exit()