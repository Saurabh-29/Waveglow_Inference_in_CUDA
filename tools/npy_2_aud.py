import os, sys
import numpy as np
from scipy.io.wavfile import write

folder = sys.argv[1]

for file in os.listdir(folder):
	if file.endswith(".npy"):
		print(file, file.split(".")[0])
		a = np.load(folder+file)
		write(folder+file.split(".")[0]+".wav", 22050, a)