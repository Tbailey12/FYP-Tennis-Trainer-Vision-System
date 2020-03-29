import numpy as np 


if __name__ == "__main__":
	candidates_3D = np.load('candidates_3D.npy', allow_pickle=True)

	print(candidates_3D[0])