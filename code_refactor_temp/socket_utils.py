#################### - socket_utils.py - ####################
'''
All socket utilities for sending/receiving messages
'''
#############################################################
import socket
import pickle
import errno

import consts as c

class Error(Exception):
	pass

# Communication error that can be raised as exception
# Message -> b'HEADER + MESSAGE'
class CommError(Error):
	def __init__(self, message):
		self.message = message

class MyMessage(object):
	def __init__(self, message_type, message):
		self.type = message_type
		self.message = message

# Receive a message from the given socket and decodes it
def receive_message(client_socket, caller):
	try:
		message_header = client_socket.recv(c.HEADER_LEN)
		bytes_recv = len(message_header)

		# If the not all the data was received, wait for the remaining header data
		if bytes_recv < c.HEADER_LEN:
			message_header += client_socket.recv(c.HEADER_LEN-bytes_recv)
			bytes_recv = len(message_header)

		# Check if valid header
		if bytes_recv != c.HEADER_LEN:
			raise CommError("Connection closed unexpectedly")

		message_len = int(message_header.decode('utf-8'))
		bytes_recv = 0
		chunks = []
		
		while bytes_recv < message_len:
			chunk = client_socket.recv(min(message_len-bytes_recv, c.CHUNK_SIZE))
			
			# if the whole message cannot be receieved
			if chunk == b'':
				raise RuntimeError('Socket connection broken')

			chunks.append(chunk)
			bytes_recv += len(chunk)

		message_b = b''.join(chunks)
		message = pickle.loads(message_b)
		return {"header": message_header, "data": message}

	except IOError as e:
		print(e)
		pass
	except ConnectionResetError as e:
		print(e)
		pass
	except Exception as e:
		print(e)
		pass

# Encodes the given message and sends it
def send_message(client_socket, message, caller):
	message = pickle.dumps(message)
	message_len = len(message)

	message_header = f"{message_len:<{c.HEADER_LEN}}".encode('utf-8')
	message = bytes(message_header) + message

	message_len = len(message)
	bytes_sent = 0
	try:
		while bytes_sent < message_len:
			chunk_sent = client_socket.send(message[bytes_sent:])
			bytes_sent += chunk_sent

			if chunk_sent == 0:
				raise RuntimeError("Socket connection broken")

		return True	# message sent successfully

	except ConnectionResetError as e:
		# close the client if the server disconnects
		if caller == c.CLIENT_NAME:
			raise CommError("Error: Server disconnected") from e
		elif caller == c.SERVER_NAME:
			return None

	except IOError as e:
		# when there is too many bytes for socket send buffer
		if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
			print(f"Send Error: {str(e)}")
			if caller == c.CLIENT_NAME:  # close the client if the server has disconnected
				raise CommError("Error: Server disconnected") from e
			elif caller == c.SERVER_NAME:  # return None if the client has disconnected
				return None
		print(e)
		return None  # if there are no more messages and no errors

	except Exception as e:
		print("Send Error", e)
		raise CommError("Send Error") from e