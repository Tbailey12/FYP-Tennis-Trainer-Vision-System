'''
socket tutorial (SentDex) https://www.youtube.com/watch?v=ytu2yV3Gn1I
'''

import socket
import select
import time
import sys

import consts as c
import socket_funcs as sf

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # create IPV4 socket for server
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # allows us to reconnect to same port

server_socket.bind((c.IP, c.PORT))
server_socket.listen()

print("Server started")

sockets_list = [server_socket]  # list of sockets, init with server socket
clients = {}  # list of clients

def print_debug(my_print):
    if c.DEBUG:
        print(my_print)

# def receive_message(client_socket):
#     try:
#         message_header = client_socket.recv(c.HEADER_LENGTH)
#
#         if not len(message_header):  # if no data, the client closed the connection
#             return False
#
#         message_length = int(message_header.decode("utf-8").strip())  # obtain the message length from the header
#         return {"header": message_header, "data": client_socket.recv(message_length)}  # return raw bytes of message
#     except:  # if the client disconnects for some reason
#         return False


# def send_message(use_socket, message):
#     message = message.encode('utf-8')
#     message_header = f"{len(message):<{c.HEADER_LENGTH}}".encode('utf-8')  # create message header
#     try:
#         use_socket.send(message_header + message)
#     except ConnectionResetError as e:
#         return
#     print_debug(f"Sent message to server: {message.decode('utf-8')}")


while True:
    time.sleep(1)
    for client_socket in sockets_list:
        if client_socket != server_socket:  # if the socket is not the server socket
            # send left message
            if clients[client_socket]['data'] == c.LEFT_CLIENT:
                print_debug("Sending message to left client")
                sf.send_message(client_socket, 5)

    # syntax for select.select()
    # (sockets we read, sockets we write, sockets that error)
    read_sockets, _, exception_sockets = select.select(sockets_list, [], sockets_list, 0)

    # if any of the read_sockets have new data
    for notified_socket in read_sockets:
        # new client connected
        if notified_socket == server_socket:  # client has connected, so accept and handle connection
            client_socket, client_address = server_socket.accept()

            client = sf.receive_message(client_socket)
            if client is None:  # client disconnected while sending
                continue
            sockets_list.append(client_socket)  # append new socket to list of client sockets
            clients[client_socket] = client
            print(
                f"Accepted new connection from {client_address[0]}:{client_address[1]}, client:{client['data']}")
        # existing client connected
        else:
            message = sf.receive_message(notified_socket)

            if message is None:
                print(f"Closed connection from {clients[notified_socket]['data']}")
                sockets_list.remove(notified_socket)
                del clients[notified_socket]
                continue

            client = clients[notified_socket]
            print_debug(f"Received message from {client['data']}: {message['data']}")

    # if there is an exception, remove the socket from the list
    for notified_socket in exception_sockets:
        sockets_list.remove(notified_socket)
        del clients[notified_socket]
