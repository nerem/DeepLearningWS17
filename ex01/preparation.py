import numpy as np
from scipy.misc import imread

# Fromats the data of a pair of images (original, vignetted) for use in our model.
# Input the image names.
def prep(img_name, img_vignetted_name, n):

	img = imread(img_name)
	img_vignetted = imread(img_vignetted_name)

	# Get image dimensions.
	height = img.shape[0]
	width = img.shape[1]

	# Create vector of radii powers [r, r**2, ..., r**n] for each pixel.
	x, y = np.meshgrid(np.arange(width) - width/2, np.arange(height) - height/2)
	r = (np.sqrt(x ** 2 + y ** 2) / np.sqrt((width/2) ** 2 + (height/2) ** 2)).flatten()
	mg_r, mg_p = np.meshgrid(r, np.arange(1,n+1))
	R = mg_r ** mg_p

	# Separate and flatten the rgb channels of the original image.
	y_r = np.array(img[:,:,0]).flatten()
	y_g = np.array(img[:,:,1]).flatten()
	y_b = np.array(img[:,:,2]).flatten()

	# Add initial color data as first component to training data ([y, r, r**2, ..., r**n]).
	R = np.append([np.concatenate([y_r, y_g, y_b])], np.tile(R, 3), axis = 0).T

	# Separate and flatten the rgb channels of the vignetted image.
	yv_r = np.array(img_vignetted[:,:,0]).flatten()
	yv_g = np.array(img_vignetted[:,:,1]).flatten()
	yv_b = np.array(img_vignetted[:,:,2]).flatten()

	# Concatenate the output color data.
	y = np.concatenate([yv_r, yv_g, yv_b])

	return np.array(R, dtype = np.float32), np.array(y, dtype = np.float32, ndmin = 2).T

# Prepare multiple images at once using prep.
# Input arrays of image names.
def multi_prep(imgs, imgs_vignetted, n):

	R = np.empty((0, n + 1), dtype = np.float32)
	y = np.empty((0, 1), dtype = np.float32)
	for i in range(len(imgs)):
		tmp_R, tmp_y = prep(imgs[i], imgs_vignetted[i], n)
		R = np.vstack((R, tmp_R))
		y = np.vstack((y, tmp_y))

	return R, y

# Split data into training and testing sets at a given ratio.
def split(x, y, ratio = 0.8):

	# splitting index (rounded in favor of training data)
	s = int(x.shape[0] * ratio) + 1

	# Create and randomly permute an index.
	index = np.zeros(x.shape[0], dtype = bool)
	index[:s] = True;
	index = np.random.permutation(index)

	return x[index], y[index], x[np.logical_not(index)], y[np.logical_not(index)]
