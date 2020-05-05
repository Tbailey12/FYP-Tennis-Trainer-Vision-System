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

def create_client_socket():
	client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # create IPV4 socket
	client_socket.settimeout(c.SOCKET_TIMEOUT)
	return client_socket

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
			break

if __name__ == "__main__":
	client_name = c.LEFT_CLIENT
	if len(sys.argv)>1:
		client_name = sys.argv[1]
		print(client_name)

	print(f"Client: {client_name} started")
	client_socket = create_client_socket()
	connect_to_server(client_socket, client_name)
	while(True):
		time.sleep(1)