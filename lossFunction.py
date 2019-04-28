import numpy as np 
import 
import keras.backend as K 

def smoothL1(y_pred, y_truth): 
	"""
		Calculate the loss between predicted bboxes and ground truth
		Cite: Faster R-CNN
		Input Dim: (batch_sizes, n_box_total, 4)
	"""
	thres = K.variable(1)
	diff = K.abs(y_pred - y_truth)
	
	loss = K.switch(diff < thres, 0.5 * diff ** 2, K.sum(diff, -0.5))

	return K.sum(loss, axis=-1)

def confLoss(y_pred, y_truth): 
	"""
		Calculate the softmax loss over multiple classes confidences
		Cite: SSD
		Input Dim: (batch_sizes, n_box_total, n_classes)
			y_pred: softmax confidence scores over all classes
			y_truth: ground truth label (one-hot encoded)
	"""
	

def intersection(default, y_truth):
	"""
		Input: 
			- default: a 2D tensor of shape (n_default, 4)
						with A the number of default boxes at 
						each pixel location. 
			- y_truth: a 2D tensor of shape (n_truth, 4)
						with B the number of ground-truth boxes. 

		Output: 
			intersect: a 2D tensor of shape (n_default, n_truth), returning the
						intersection area of each default boxes for
						every ground-truth boxes. 

	""" 
	# Get the number of default boxes and ground-truth
	n_default = default.shape[0]
	n_truth = y_truth.shape[0]
	# Expand to 3D 
	default = np.expand_dims(default, axis=1)  # (n_default, 1, 4)
	y_truth = np.expand_dims(y_truth, axis=0)  # (1, n_truth, 4)

	# Resize the arrays to dimension (n_default, n_truth, 4)
	default = np.resize(default, (n_default, n_truth, 4))
	y_truth = np.resize(y_truth, (n_default, n_truth, 4))

	# Find the right coordinates of the intersection
	right_xy = np.amin(default[:, :, 2:], y_truth[:, :, 2:], axis=3)
	# Find the left coordinates of the intersection
	left_xy = np.amax(default[:, :, :2], y_truth[:, :, :2], axis=3)

	# Calculate width and height of intersection
	# set negative values to 0 (no intersecting area)

	intersection_wh = np.clip(right_xy - left_xy, a_min=0)

	return intersection_wh[:, :, 0] * intersection_wh[:, :, 1]

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

	intersect = intersection(default, y_truth)

	# Expand to 3D 
	default = np.expand_dims(default, axis=1)
	y_truth = np.expand_dims(y_truth, axis=0)

	# Resize the arrays to dimension (n_default, n_truth, 4)
	default = np.resize(default, (n_default, n_truth, 4))
	y_truth = np.resize(y_truth, (n_default, n_truth, 4))

	# Calculate areas of every default boxes and ground-truth boxes
	area_default = (default[:, :, 2] - default[:, :, 0])*(default[:, :, 3] - default[:, :, 1])
	area_truth = (y_truth[:, :, 2] - y_truth[:, :, 0])*(y_truth[:, :, 3] - y_truth[:, :, 1])

	# Union of area

	union = area_default + area_truth - intersect

	return intersect/union




