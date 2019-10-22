import socket
import select
import pickle

import consts as c

'''
pickles and encodes data with a with a defined 'data_type' in the format
typeheader -> data_type -> messageheader -> message
'''


def send_message(socket, data_type, message_data):
    # format type header and encode type
    message_type = data_type.encode('utf-8')
    type_header = f"{len(message_type):<{c.HEADER_LENGTH}}".encode('utf-8')

    # format message header and encode message
    message = pickle.dumps(message_data)
    message_header = f"{len(message):<{c.HEADER_LENGTH}}".encode('utf-8')

    # construct message
    message = bytes(type_header + message_type + message_header) + message

    try:
        socket.send(message)
    except Exception as e:  # could not send message for some reason
        print("Error", e)
        return False
    return True  # if message sent successfully


'''
receives a message from the given socket and decodes it
'''


def receive_message(client_socket):
    type_header = client_socket.recv(c.HEADER_LENGTH)
    if not len(type_header):  # if something goes wrong, return None
        print("Connection closed unexpectedly")
        return None
    type_length = int(type_header.decode('utf-8').strip())
    message_type = client_socket.recv(type_length).decode('utf-8')

    message_header = client_socket.recv(c.HEADER_LENGTH)
    message_length = int(message_header.decode('utf-8'))
    message = pickle.loads(client_socket.recv(message_length))

    return {"type": message_type, "data": message}
