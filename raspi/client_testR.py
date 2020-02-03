# clienttest
import time
import numpy as np
import matplotlib.pyplot as plt

import client
import consts as c
import socket_funcs as sf

import sys

frames = 90

if __name__ == "__main__":
## -- setup client connection to server -- ##
	client.connect_to_server(name=c.RIGHT_CLIENT)
	my_message = []

	for i in range(0,1000):
		my_message.append((i,i,i,i,i,i,i,i,i,i))

	start = time.time()
	for i in range(0,frames):
		print(i)
		sent = sf.send_message(client.client_socket, (i,my_message), c.CLIENT)
		if not sent:
			print("Not sent")
			
	print(f"time: {time.time()-start}")
	quit()