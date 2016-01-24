from PIL import Image
import glob

for filename in glob.glob("*.jpg"):
	img = Image.open(filename).convert('L')
	img.show()
	img.save(filename)


