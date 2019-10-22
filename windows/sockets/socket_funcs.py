import socket
import select
import pickle

import consts as c

'''
pickles and encodes data with a with a defined 'data_type' in the format
typeheader -> data_type -> messageheader -> message
'''


def send_message(socket, message_data):
    # format type header and encode type
    # message_type = data_type.encode('utf-8')
    # type_header = f"{len(message_type):<{c.HEADER_LENGTH}}".encode('utf-8')

    # format message header and encode message
    message = pickle.dumps(message_data)
    message_header = f"{len(message):<{c.HEADER_LENGTH}}".encode('utf-8')

    # construct message
    message = bytes(message_header) + message

    try:
        socket.send(message)
    except Exception as e:  # could not send message for some reason
        print("Send Error", e)
        return False
    return True  # if message sent successfully


'''
receives a message from the given socket and decodes it
'''


def receive_message(client_socket):
    try:
        message_header = client_socket.recv(c.HEADER_LENGTH)
        if not len(message_header):  # if something goes wrong, return None
            print("Connection closed unexpectedly")
            return None
        message_length = int(message_header.decode('utf-8'))
        message = pickle.loads(client_socket.recv(message_length))
    except Exception as e:
        # print("Recv Error",e)
        return None

    return {"header": message_header, "data": message}
