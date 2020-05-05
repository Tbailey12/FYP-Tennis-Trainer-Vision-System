#################### - client.py - ####################
'''
Functions for client communication with server
'''
#######################################################

import socket
import select
import time
import sys

import consts as c
import socket_utils

# Creates the client socket for communication with the server
def create_client_socket():
	client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # create IPV4 socket
	client_socket.settimeout(c.SOCKET_TIMEOUT)
	return client_socket

# Connects to the server and sends client data to server
def connect_to_server(client_socket, client_name):
	# Attempt to connect to the server, if connection is refused wait 1s then try again
	while True:
		try:
			client_socket.connect((c.SERVER_IP, c.PORT))
			client_socket.setblocking(True)
			
			socket_utils.send_message(client_socket, client_name, c.CLIENT_NAME)
			print(f"Connection established to server on {c.SERVER_IP}:{c.PORT}")
			return True

		except (socket.error, socket.timeout) as e:
			print(f"Connection could not be established to server on {c.SERVER_IP}:{c.PORT}, trying again...")
			time.sleep(1)
			continue

		except Exception as e:
			print(f"Exception occurred: {e}")
			raise Exception(e)
			break

def read_server_messages(client_socket):
	message_list = []
	try:
		read_sockets, _, exception_sockets = select.select([client_socket], [], [client_socket], 0)

		while read_sockets:
			for notified_socket in read_sockets:
				message = socket_utils.receive_message(notified_socket, c.CLIENT_NAME)
				if message is None:
					raise socket_utils.CommError(f"Connection closed from {c.SERVER_NAME}")
				else:
					message_list.append({"client": c.SERVER_NAME, "data": message['data']})

			read_sockets, _, exception_sockets = select.select([client_socket], [], [client_socket], 0)
		return message_list

	except socket_utils.CommError as e:
		raise socket_utils.CommError(e) from e

if __name__ == "__main__":
	client_name = c.LEFT_CLIENT
	if len(sys.argv)>1:
		client_name = sys.argv[1]
		print(client_name)

	print(f"Client: {client_name} started")
	client_socket = create_client_socket()

	connect_to_server(client_socket, client_name)
	
	while True:
		message = socket_utils.MyMessage(c.TYPE_STR, f"Hello server: {time.time()}")

		sent = socket_utils.send_message(client_socket, message, c.CLIENT_NAME)
		print(f"message sent to server: {sent}")

		message_list = read_server_messages(client_socket)
		for message in message_list:
			print(message['data'].message)

		time.sleep(1)