#################### - server.py - ####################
'''
All sever functions and main server code
'''
#######################################################
import socket
import select
import time

import consts as c
import shared_funcs as sf
import socket_utils

# Add a new client to the client_dict
def add_new_client(server_socket):
	client_socket, client_address = server_socket.accept()
	try:
		client = socket_utils.receive_message(client_socket, c.SERVER_NAME)
		if client is not None:
			socket_list.append(client_socket)
			client_dict[client_socket] = client['data']
			print(f"Accepted connection from client: {client['data']} on {client_address[0]}:{client_address[1]}")
			return True
		else:
			return False

	except socket_utils.CommError as e:
		print(e.message)
		return False

# Read all incoming server messages
def read_client_messages():
	message_list = []

	# select.select(rlist, wlist, xlist, timeout=...)
	read_sockets, _, exception_sockets = select.select(socket_list, [], socket_list, 0)

	if read_sockets:
		for notified_socket in read_sockets:
			if notified_socket == server_socket:
				add_new_client(server_socket)
			else:
				try:
					message = socket_utils.receive_message(notified_socket, c.SERVER_NAME)
					if message is not None:
						client = client_dict[notified_socket]
						message_list.append({"client": client['data'], "data": message['data']})
						# sf.print_debug(f"Received message from {client['data']}: {message['data']}")
					else:
						raise socket_utils.CommError(f"client: {client_dict[notified_socket]} disconnected")

				except socket_utils.CommError as e:
					print(e)
					socket_list.remove(notified_socket)
					del client_dict[notified_socket]
					continue

		return message_list

	else:
		return []

# def check_client_connection(client_socket):

# Wait for both left and right client to connect to the server
def initialise_server():
	while True:
		read_client_messages()
		if len(client_dict) >= 2:
			left_client_connected, right_client_connected = False, False

			for client_socket in client_dict:
				left_client_connected = True if client_dict[client_socket] == c.LEFT_CLIENT else left_client_connected
				right_client_connected = True if client_dict[client_socket] == c.RIGHT_CLIENT else right_client_connected
			if left_client_connected and right_client_connected:
				print('both clients connected')
				return True

		time.sleep(0.01)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)	# create IPV4 socket for server

server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)	# allows reconnection to same port
server_socket.bind((c.SERVER_IP, c.PORT))
server_socket.listen()

socket_list = [server_socket]
client_dict = {}

if __name__ == "__main__":
	print(f"Server started on {c.SERVER_IP}:{c.PORT}")
	initialise_server()