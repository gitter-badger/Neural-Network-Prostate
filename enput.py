from PIL import Image
import numpy as np
import glob, os

for img in glob.glob("*.jpg"):
	with open("Prostate_Cancer_Data1", "wb") as f:
	  	for img in glob.glob("*.jpg"):
			label = np.array([0], dtype = np.uint8)
			f.write(label.tostring())
			im = np.array(Image.open(img), dtype = np.uint8)
			f.write(im[:, :, 0].tostring())
			f.write(im[:, :, 1].tostring())
			f.write(im[:, :, 2].tostring())
			print(im)
