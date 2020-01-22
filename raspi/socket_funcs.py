import socket
import select
import pickle
import errno
import sys

import consts as c

debug = c.DEBUG


def print_debug(my_print):
    if debug:
        print(my_print)


'''
pickles and encodes data with a with a defined 'data_type' in the format
typeheader -> data_type -> messageheader -> message
'''


def send_message(socket, message_data, caller):
    # format type header and encode type
    # message_type = data_type.encode('utf-8')
    # type_header = f"{len(message_type):<{c.HEADER_LENGTH}}".encode('utf-8')

    # format message header and encode message
    message = pickle.dumps(message_data)
    message_header = f"{len(message):<{c.HEADER_LENGTH}}".encode('utf-8')
    # construct message
    message = bytes(message_header) + message

    try:
        socket.sendall(message)
    except ConnectionResetError as e:
        if caller == c.CLIENT:  # close the client if the server has disconnected
            print('Error: Server has disconnected, closing client')
            sys.exit()
        elif caller == c.SERVER:  # return None if the client has disconnected
            return None
        return None
    except IOError as e:
        # when there is too many byes for socket send buffer
        if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
            print_debug(f"Send Error:{str(e)}")
            if caller == c.CLIENT:  # close the client if the server has disconnected
                print('Error: Server has disconnected, closing client')
                sys.exit()
            elif caller == c.SERVER:  # return None if the client has disconnected
                return None
        return None  # if there are no more messages and no errors
    except Exception as e:  # could not send message for some reason
        print("Send Error", e)
        return False
    return True  # if message sent successfully


'''
receives a message from the given socket and decodes it
'''


def receive_message(client_socket, caller):
    try:
        message_header = client_socket.recv(c.HEADER_LENGTH)
        if not len(message_header):  # if something goes wrong, return None
            print("Connection closed unexpectedly")
            return None
        message_length = int(message_header.decode('utf-8'))
        received = client_socket.recv(message_length)
        while len(received) < message_length:
            bytes_remaining = message_length - len(received)
            received += client_socket.recv(bytes_remaining)
        message = pickle.loads(received)
        return {"header": message_header, "data": message}

    except IOError as e:
        # errors when there are no more messages to be received
        if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
            print_debug(f"Read Error:{str(e)}")
            if caller == c.CLIENT:  # close the client if the server has disconnected
                print('Error: Server has disconnected, closing client')
                sys.exit()
            elif caller == c.SERVER:  # return None if the client has disconnected
                return None
        return None  # if there are no more messages and no errors
    except ConnectionResetError as e:
        if caller == c.CLIENT:  # close the client if the server has disconnected
            print('Error: Server has disconnected, closing client')
            sys.exit()
        elif caller == c.SERVER:  # return None if the client has disconnected
            return None
    except Exception as e:
        print("Recv Error", e)  # could not receive message for some reason
        return None
