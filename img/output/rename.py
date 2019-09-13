import os

folder = "binary_image_search"

root = os.getcwd()
os.chdir(folder)

for i, file in enumerate(os.listdir()):
	os.rename(file, str("%04d" % i)+".png")