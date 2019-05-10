import numpy as np 
import itertools as it 
from math import sqrt

def generate_default_boxes(n_layers=6, 
							min_scale=0.2,
							max_scale=0.9,
							map_size=[38, 19, 10, 5, 3, 1], 
							aspect_ratios=[0.5, 1, 2]): 

	"""
		Output: 
			- default: a 2D array (#defaults, 4) containing the coordinates [x, y, h, w] 
						of all default boxes relative to the image size. 
	"""
	scales = np.linspace(start=min_scale, stop=max_scale, num=n_layers)

	default = np.empty(shape=(0, 4))

	for level in range(n_layers):

		scale = scales[level] 
		# For each pixel location in the feature map
		for i, j in it.product(range(map_size[level]), repeat=2): 
			
			# Calculate the center of each default box
			x = (i + 0.5)/map_size[level]
			y = (j + 0.5)/map_size[level]

			for ratio in aspect_ratios: 
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

def IoU(default, 
		boxes): 
	"""
		Input: 
			- default: an array of coordinates of default boxes
			- boxes   : an array of coordinates of ground-truth boxes
		Output: 
			- iou:  a 2D tensor of shape (n_default, n_truth), returning the
						Jaccard index of each default boxes for
						every ground-truth boxes. 
	"""

	x1, y1, w1, h1 = np.split(default, 4, axis=1)
	x2, y2, w2, h2 = np.split(boxes, 4, axis=1)

	x12 = x1 + w1
	x22 = x2 + w2
	y12 = y1 + h1
	y22 = y2 + h2

	n_default = default.shape[0]
	n_truth = boxes.shape[0]

	topleft_x = np.maximum(x1,np.transpose(x2))
	topleft_y = np.maximum(y1,np.transpose(y2))

	botright_x = np.minimum(x12,np.transpose(x22))
	botright_y = np.minimum(y12,np.transpose(y22))

	intersect = (botright_x - topleft_x)*(botright_y - topleft_y)

	# Calculate areas of every default boxes and ground-truth boxes
	area_default = w1*h1
	area_truth = w2*h2

	# Union of area
	union = area_default + np.transpose(area_truth) - intersect

	# Avoid division by 0
	union = np.maximum(union, 1e-18)

	iou_matrix = np.maximum(intersect/union, 0)

	return iou_matrix