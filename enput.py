import numpy as np
import cPickle as pk
from PIL import Image

import glob


def loadImage(filename):
	'''return a Image obect'''
	return Image.open(filename)


def pickle_data(filename, data, mode='wb'):
	with open(filename, mode) as file:
		pk.dump(data, file)


def unpickle_data(filename, mode = 'rb'):
	with open(filename, mode) as file:
		data = pk.load(file)
	return data

def loadAllPic():
	dict = {}
	imgdata = []
	imglabel = []

	for file in glob.glob('*.jpg'):
		print file
		img = loadImage(file)
		rawdata = img.load()
		redchannel = [rawdata[x, y][0] for x in range(img.width) for y in range(img.height)]
		greenchannel = [rawdata[x, y][1] for x in range(img.width) for y in range(img.height)]
		bluechannel = [rawdata[x, y][2] for x in range(img.width) for y in range(img.height)]
		nparray = np.array(redchannel + greenchannel + bluechannel)
		imgdata.append(nparray)
		imglabel.append("Gleason_3")
	dict['data'] = imgdata
	dict['labels'] = imglabel
	return dict


def main():
	dict = loadAllPic()
	pickle_data('Prostate_Cancer_Data.binary', dict)

	data = unpickle_data('Prostate_Cancer_Data.binary')
	print data.viewkeys()
	print data['labels'][:100]
	print data['data'][0]



if __name__ == '__main__':
	main()
