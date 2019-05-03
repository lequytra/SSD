import numpy as np 
from math import sqrt
import tensorflow as tf 
import keras.backend as K 
import itertools as it

from lossFunction import IoU

class Encoder(): 
	def __init__(self, 
				numClasses=10,
				input_shape=(300,300,3),
				min_scale=0.2, 
				max_scale=0.9, 
				aspect_ratio=[0.5, 1, 2], 
				n_predictions=6, 
				prediction_size=[38, 19, 10, 5, 3, 1]): 
	""" 
		Input: 
			- input_shape: the shape of the input image
			- numClasses: the number of classes that will be trained
			- min_scale: the smallest feature map scale
			- max_scale: the largest feature map scale
			- aspect_ratio: a list of aspect ratios for each default boxes
			- n_predictions: number of prediction layers
			- prediction_size: a list of sizes for the predictions
								The number of element must be equal to n_predictions
	"""

		self.im_height = input_shape[0]
		self.numClasses = numClasses
		self.im_width = input_shape[1]
		self.n_layers = n_predictions
		# Calculate the scale at each prediction layer
		self.scales = np.linspace(start=min_scale, stop=max_scale, num=n_predictions)
		self.map_size = prediction_size
		self.default = np.empty(shape=(1, 4))
		self.background_id = 0



	def generate_default_boxes(): 

	"""
		Output: 
			- default: a 2D array (#defaults, 4) containing the coordinates [x, y, h, w] 
						of all default boxes relative to the image size. 
	"""

		for level in range(self.n_layers):

			scale = self.scales[level] 
			# For each pixel location in the feature map
			for i, j in it.product(range(self.map_size[level]), repeat=2): 
				
				# Calculate the center of each default box
				x = (i + 0.5)/self.map_size[level]
				y = (j + 0.5)/self.map_size[level]

				for ratio in aspect_ratio: 
					box = np.empty(shape=(1, 4))

					box[:, 0] = x
					box[:, 1] = y

					# Calculate the width and height
					w = scale*sqrt(ratio)
					h = scale/sqrt(ratio)

					box[:, 2] = w
					box[:, 3] = h

					self.default = np.concatenate((default, box), axis=0)

		return self.default

	def encode_format():
		"""

		Output: 
			- encoded: an array with the shape (#default, numClasses + 1 + 4)
						The content of the array is trivial, all is first set
						to background class
						The last 4 values contains the coordinates of the 
							corresponding default boxes
		"""
		n_default = self.default.shape[0]
		# Create the 2D array to hold data
		encoded = np.zeros(shape=(n_default, self.numClasses + 1 + 4))
		# Set all to back ground class
		encoded[:, self.background_id] = 1
		# Set the last 4 values to contains the coordinates of the default boxes
		encoded[:, -4:] = self.default

		return encoded

	def encode_label(y_truth): 
		"""
			Input: 
				- y_truth: a 2D array (#n_boxes, 5) containing the coordinates and class of 
							the ground-truth labels coordinates must be normalized [0, 1]
		"""

		# Calculate the IoU of the default boxes with all the grouth-truth boxes
		# The result is a 2D matrix with the shape (#default, #n_boxes)
		iou_matrix = IoU(self.default, y_truth)
		# Find #default and n_boxes
		n_default = self.default.shape[0]
		n_boxes = y_truth.shape[0]
		# For each gt, find the default box index that has the best match
		max_indices = np.argmax(iou_matrix, axis=0)

		assert max_indices.shape == n_boxes

		# create a 2D zero matrix for one-hot encode (n_boxes, numClasses + 1)
		one_hot = np.zeros(shape=(n_boxes, numClasses + 1))

		# Obtain the boxes' labels
		labels = y_truth[:, -1]

		# Encode the classes






