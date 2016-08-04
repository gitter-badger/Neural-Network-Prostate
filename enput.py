from PIL import Image
import numpy as np
import glob


for img in glob.glob("*.jpg"):
	for i in range(1):
		im = Image.open(img)
		im = (np.array(im))
		print(im)

		r = im[:,:,0].flatten()
		g = im[:,:,1].flatten()
		b = im[:,:,2].flatten()
		label = [0]

		begin = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)

	new_array = np.array(list(r) + list(g) + list(b), np.uint8)
	NEW = np.append(begin, new_array, 0)
	NEW.tofile("Prostate_Cacer_Data1.bin")
