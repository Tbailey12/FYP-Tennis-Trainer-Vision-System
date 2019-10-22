import socket
import select
import errno
import sys
import time
from datetime import datetime

HEADER_LENGTH = 10
IP = "127.0.0.1"
PORT = 1234

this_client = "left"

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # create IPV4 socket for client

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
    break
client_socket.setblocking(False)

client_name = this_client.encode('utf-8')
client_name_header = f"{len(client_name):<{HEADER_LENGTH}}".encode('utf-8')  # create header
client_socket.send(client_name_header + client_name)  # send name to server

while True:
    time.sleep(1)
    message = f"Time:{datetime.now()}"  # message to be sent to the server

    if message:
        message = message.encode('utf-8')
        message_header = f"{len(message):<{HEADER_LENGTH}}".encode('utf-8')  # create message header
        client_socket.send(message_header + message)  # send message to server

    # receive messages
    try:
        while True:
            client_name_header = client_socket.recv(HEADER_LENGTH)
            if not len(client_name_header):
                print("Connection closed by the server")
                sys.exit()

            client_name_length = int(client_name_header.decode('utf-8').strip())  # decode the client name length
            client_name = client_socket.recv(client_name_length.decode('utf-8'))  # receive the client name

            message_header = client_socket.recv(HEADER_LENGTH)
            message_length = int(message_header.decode('utf-8').strip())
            message = client_socket.recv(message_length.decode('utf-8'))

            print(f"{client_name}:{message}")

    except IOError as e:
        # errors when there are no more messages to be received
        if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
            print('Read error:', str(e))
            sys.exit()
        continue  # if there are no more messages and no errors

    except Exception as e:
        print('General error', str(e))
        sys.exit()
