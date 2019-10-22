import socket
import select
import errno
import sys
import time
from datetime import datetime

DEBUG = True

TYPE_TEXT = "text"
TYPE_VAR = "var"
HEADER_LENGTH = 10
IP = "127.0.0.1"
PORT = 1234

name = "left"

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # create IPV4 socket for client


def print_debug(my_print):
    if DEBUG:
        print(my_print)


def send_message(use_socket, message):
    message = message.encode('utf-8')
    message_header = f"{len(message):<{HEADER_LENGTH}}".encode('utf-8')  # create message header
    try:
        use_socket.send(message_header + message)
    except ConnectionResetError as e:
        print("Error", e)
        sys.exit()
    print_debug(f"Sent message to server: {message.decode('utf-8')}")


def connect_to_server():
    # attempt to connect to the server, if connection refused, wait 1s then try again
    while True:
        try:
            client_socket.connect((IP, PORT))  # connect to server
        except socket.error as e:
            print("Connection could not be established to server, trying again...")
            time.sleep(1)
            continue
        except Exception as e:
            print('Error', str(e))
            sys.exit()
        print(f"Connection Established to server on {IP}:{PORT}")
        client_socket.setblocking(False)
        send_message(client_socket, name)  # send name to server
        break


def receive_message():
    # receive messages
    try:
        while True:
            # client_name_header = client_socket.recv(HEADER_LENGTH)
            # if not len(client_name_header):
            #     print("Connection closed by the server")
            #     sys.exit()

            # client_name_length = int(client_name_header.decode('utf-8').strip())  # decode the client name length
            # client_name = client_socket.recv(client_name_length.decode('utf-8'))  # receive the client name

            message_header = client_socket.recv(HEADER_LENGTH)
            if not len(message_header):
                print("Connection closed by the server")
                sys.exit()
            message_length = int(message_header.decode('utf-8').strip())
            message = client_socket.recv(message_length).decode('utf-8')

            print_debug(f"Received message from Server: {message}")
            return message

    except IOError as e:
        # errors when there are no more messages to be received
        if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
            print('Read error:', str(e))
            sys.exit()
        return None  # if there are no more messages and no errors

    # except Exception as e:
    #     print('General error', str(e))
    #     sys.exit()


if __name__ == "__main__":
    connect_to_server()
    while True:
        time.sleep(1)
        send_message(client_socket, f"Time:{datetime.now()}")  # send message to server
        message_recv = receive_message()  # receive messages from the server
