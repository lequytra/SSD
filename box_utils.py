import numpy as tf 
import tensorflow as tf
from generatingBoxes import generate_default_boxes
import keras.backend as K 

def IoU(default, y_truth): 
	"""
		Input: 
			- default: a 2D tensor of shape (n_default, 4)
						with A the number of default boxes at 
						each pixel location. 
			- y_truth: a 2D tensor of shape (n_truth, 4)
						with B the number of ground-truth boxes. 

		Output: 
			- iou:  a 2D tensor of shape (n_default, n_truth), returning the
						Jaccard index of each default boxes for
						every ground-truth boxes. 
	"""

	x1, y1, w1, h1 = tf.split(default, 4, axis=1)
	x2, y2, w2, h2 = tf.split(y_truth, 4, axis=1)

	x12 = x1 + w1
	x22 = x2 + w2
	y12 = y1 + h1
	y22 = y2 + h2

	n_default = default.shape[0]
	n_truth = y_truth.shape[0]

	topleft_x = tf.maximum(x1,tf.transpose(x2))
	topleft_y = tf.maximum(y1,tf.transpose(y2))

	botright_x = tf.minimum(x12,tf.transpose(x22))
	botright_y = tf.minimum(y12,tf.transpose(y22))

	intersect = (botright_x - topleft_x)*(botright_y - topleft_y)

	# Calculate areas of every default boxes and ground-truth boxes
	area_default = w1*h1
	area_truth = w2*h2

	# Union of area

	union = area_default + area_truth - intersect

	return tf.maximum(intersect/union, 0)


## TEST: 
# default1 = generate_default_boxes()
# iou_matrix = IoU(default1, default1)
# n_default = default1.shape[0]



