class myclass():
	def __init__(self, a):
		self.a = a

	def add_one(self):
		self.a.append(1)
		print(a)

if __name__ == "__main__":
	a = []
	obj = myclass(a)
	print(a)
	obj.add_one()
	print(a)


# w,h = (1280,720)
# w,h = (640,480)
# resolution = w,h
# framerate = 90

# def print_l(print_lock, message):
#     print_lock.acquire()
#     try: 
#         print(message)
#     finally:
#         print_lock.release()

# def ImageProcessor(unprocessed_frames, print_lock, process_run):
#     # stream = io.BytesIO()
#     # self.event = multiprocessing.Event()

#     while process_run:
#         try:
#             # attempts to get a new frame, raises exception if none available
#             # print(unprocessed_frames.qsize())

#             frame_num, frame_buf = unprocessed_frames.get_nowait()
#             # y_data = np.frombuffer(frame_buf, dtype=np.uint8, count=w*h).reshape((h,w))
#             # print(y_data)
#             # stream.write(frame)
#             # stream.seek(0)
#             # Read the image and do some processing on it

#             # im = Image.open(stream) # read image from stream
#             # img = np.array(im) # convert image to numpy array

#             # print_l(print_lock,img)
#             # print_lock.acquire()
#             time.sleep(0.02)

#             # cv2.imwrite(f'{frame_num:03d}.png',y_data)
#             # cv2.imwrite(f'{frame_num:03.3f}.png',img)
#             # print_lock.release()
#             # Reset the stream
#             # stream.seek(0)
#             # stream.truncate()

#         except queue.Empty:
#             if process_run.value is False:
#                 break
#             continue
#     return
# class ProcessOutput(object):
#     def __init__(self):
#         self.number_of_processors = 4
#         self.unprocessed_frames = multiprocessing.Queue()
#         self.processes = []
#         self.print_lock = multiprocessing.Lock()
#         self.frame_num = 0
#         self.start = time.time()
#         self.done = False
#         self.process_run = multiprocessing.Value(ctypes.c_bool,True)

#         for i in range(self.number_of_processors):
#             processor = multiprocessing.Process(target=ImageProcessor, args=(self.unprocessed_frames, self.print_lock, self.process_run))
#             self.processes.append(processor)
#             processor.start()

#     def write(self, buf):
#         if self.done is False:
#             self.unprocessed_frames.put((self.frame_num, buf)) # add the new frame to the buffer
#             self.frame_num += 1    

#     def flush(self):
#         print('closing camera')
#         # print('flushing')
#         # When told to flush (this indicates end of recording), shut
#         # down in an orderly fashion.
#         print(self.frame_num/10)
#         for p in self.processes:
#             p.terminate()
#             p.join()

# with picamera.PiCamera() as camera:
#     camera.framerate = framerate
#     camera.resolution = resolution
#     camera.vflip=True
#     camera.annotate_frame_num = True
#     camera.annotate_text_size = 160
#     camera.annotate_foreground = picamera.Color('green')
#     time.sleep(2)

#     output = ProcessOutput()
#     camera.start_recording(output, format='yuv')

#     try:   
#         camera.wait_recording(10)
#     finally:
#         output.process_run.value = False    # ensures all processes stop on next run
#         output.done = True  # stop writing frames to queue

#     while(True):
#         dead_count = 0
#         for process in output.processes:
#             if process.is_alive() is False:
#                 dead_count+=1
#         if dead_count == output.number_of_processors:
#             break

#     camera.stop_recording()