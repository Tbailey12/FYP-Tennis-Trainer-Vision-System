import socket
import select
import errno
import sys
import time
import traceback

import consts as c
import socket_funcs as sf 

def connect_to_server(client_socket, client_name):
    # attempt to connect to the server, if connection refused, wait 1s then try again
    while True:
        try:
            client_socket.connect((c.SERVER_IP, c.PORT))  # connect to server
            print(f"Connection Established to server on {c.SERVER_IP}:{c.PORT}")
            sf.send_message(client_socket, client_name, c.CLIENT)  # send name to server
            return True

        except socket.timeout as e:
            print("Connection could not be established to server, trying again...")
            continue

        except OSError as e:
            print("Connection could not be established to server, trying again...")
            pass

        except Exception as e:
            print('Error', str(e))
            raise sf.CommError("Communication Error") from e



def read_server_messages(read_all=False):
    '''
    Read all messages from the socket connected to the server

    Return: List of messages [{header: message_header, data: message_data}]
    '''
    message_list = []
    # ---    receive all messages from the server    ---- #
    try:
        read_sockets, _, exception_sockets = select.select([client_socket], [], [client_socket], 0)
        while read_sockets:
            for notified_socket in read_sockets:
                message = sf.receive_message(notified_socket, c.CLIENT)
                if message is None:
                    print(f"Closed connection from {c.SERVER}")
                    continue
                message_list.append(message)

            if read_all:
                read_sockets, _, exception_sockets = select.select([client_socket], [], [client_socket], 0)
                continue
            else:
                break
        return message_list

    except sf.CommError as e:
        raise sf.CommError("Communication Error") from e
    
def create_client_socket():
    '''
    Creates the client socket for communication with the server

    Return: client_socket
    '''
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # create IPV4 socket
    client_socket.settimeout(c.SOCKET_TIMEOUT)
    client_socket.setblocking(True)
    return client_socket

if __name__ == "__main__":
    client_name = c.LEFT_CLIENT
    if len(sys.argv)>1:
        client_name = sys.argv[1]
    print(f"Client: {client_name} started")

    client_socket = create_client_socket()
    connect_to_server(client_socket, client_name)

    while True:
        try:
            message_list = read_server_messages(read_all=True)
            for message in message_list:
                print(message['data'].message)
                new_message = sf.MyMessage(c.TYPE_STR,f"{client_name} got message: {message['data'].message}")
                sf.send_message(client_socket, new_message, c.CLIENT)

            time.sleep(1)

        except sf.CommError as e:
            print('Server disconnected, closing client')
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