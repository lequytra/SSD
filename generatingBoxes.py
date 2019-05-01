import numpy as np 
from math import sqrt
import tensorflow as tf 
import keras.backend as K 
import itertools as it

def generate_default_boxes(input_shape=(300,300,3),
							min_scale=0.2, 
							max_scale=0.9, 
							aspect_ratio=[0.5, 1, 2], 
							n_predictions=6, 
							prediction_size=[38, 19, 10, 5, 3, 1]): 

	"""
		Input: 
			- input_shape: the shape of the input image
			- min_scale: the smallest feature map scale
			- max_scale: the largest feature map scale
			- aspect_ratio: a list of aspect ratios for each default boxes
			- n_predictions: number of prediction layers
			- prediction_size: a list of sizes for the predictions
								The number of element must be equal to n_predictions
		Output: 
			- default: a numpy array of coordinates for the default boxes
						[[x, y, h, w]] relative to the image size. 
	"""

	default = np.empty(shape=(1, 4))

	# Calculate the scale at each prediction layer
	scales = np.linspace(start=min_scale, stop=max_scale, num=n_predictions)

	for level in range(n_predictions):

		scale = scales[level] 
		# For each pixel location in the feature map
		for i, j in it.product(range(prediction_size[level]), repeat=2): 
			
			# Calculate the center of each default box
			x = (i + 0.5)/prediction_size[level]
			y = (j + 0.5)/prediction_size[level]

			for ratio in aspect_ratio: 
				box = np.empty(shape=(1, 4))

				box[:, 0] = x
				box[:, 1] = y

				# Calculate the width and height
				w = scale*sqrt(ratio)
				h = scale/sqrt(ratio)

				box[:, 2] = w
				box[:, 3] = h

				default = np.concatenate((default, box), axis=0)

	return default



default = generate_default_boxes()
print(default.shape)





