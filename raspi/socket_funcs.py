import socket
import select
import pickle
import errno
import sys

import consts as c

debug = c.DEBUG

class MyMessage(object):
	def __init__(self, my_type, message):
		self.type = my_type
		self.message = message

# class CalMessage(object):
# 	def __init__(self, num_img=c.CALIB_NUM_IMG, img_delay=c.CALIB_IMG_DELAY):
# 		self.num_img = num_img
# 		self.img_delay = img_delay

def print_debug(my_print):
	if debug:
		print(my_print)

class Error(Exception):
	pass

class CommError(Error):
	def __init__(self, message):
		self.message = message


'''
pickles and encodes data with a with a defined 'data_type' in the format
typeheader -> data_type -> messageheader -> message
'''


def send_message(socket, message_data, caller):
	# format message header and encode message
	message = pickle.dumps(message_data)
	message_len = len(message)
	message_header = f"{message_len:<{c.HEADER_LENGTH}}".encode('utf-8')
	# construct message
	message = bytes(message_header) + message

	total_sent = 0
	try:
		while total_sent < len(message):
			sent = socket.send(message[total_sent:])
			if sent == 0:
				raise RuntimeError("Socket connection broken")
			total_sent = total_sent + sent
		# socket.sendall(message)
	except ConnectionResetError as e:
		if caller == c.CLIENT:  # close the client if the server has disconnected
			raise CommError("Error: Server disconnected") from e
		elif caller == c.SERVER:  # return None if the client has disconnected
			return None
		return None
	except IOError as e:
		# when there is too many byes for socket send buffer
		if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
			print(f"Send Error:{str(e)}")
			if caller == c.CLIENT:  # close the client if the server has disconnected
				raise CommError("Error: Server disconnected") from e
			elif caller == c.SERVER:  # return None if the client has disconnected
				return None
		print(e)
		return None  # if there are no more messages and no errors
	except Exception as e:  # could not send message for some reason
		print("Send Error", e)
		raise CommError("Send Error") from e
	return True  # if message sent successfully


'''
receives a message from the given socket and decodes it
'''

def receive_message(client_socket, caller):
	try:
		message_header = client_socket.recv(c.HEADER_LENGTH)
		if len(message_header) < c.HEADER_LENGTH:
			message_header += client_socket.recv(c.HEADER_LENGTH-len(message_header))

		if len(message_header) is not c.HEADER_LENGTH:
			raise CommError("Connection closed unexpectedly")

		message_length = int(message_header.decode('utf-8'))
		bytes_received = 0
		chunks = []
		while bytes_received < message_length:
			chunk = client_socket.recv(min(message_length-bytes_received,c.CHUNK_SIZE))
			if chunk == b'':
				raise RuntimeError("Socket connection broken")
			chunks.append(chunk)
			bytes_received += len(chunk)
		message_b = b''.join(chunks)
		message = pickle.loads(message_b)
		return {"header": message_header, "data": message}

	except IOError as e:
		# errors when there are no more messages to be received
		if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
			print_debug(f"Read Error:{str(e)}")
			if caller == c.CLIENT:  # close the client if the server has disconnected
				raise CommError("Error: Server disconnected") from e
			elif caller == c.SERVER:  # return None if the client has disconnected
				return None
		return None  # if there are no more messages and no errors
	except ConnectionResetError as e:
		if caller == c.CLIENT:  # close the client if the server has disconnected
			print('Error: Server has disconnected, closing client')
			raise CommError("Recv Error: Server disconnected") from e
		elif caller == c.SERVER:  # return None if the client has disconnected
			return None
	except Exception as e:
		print("Recv Error", e)  # could not receive message for some reason
		raise CommError("Recv Error") from e
		return None