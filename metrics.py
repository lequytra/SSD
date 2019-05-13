import keras.backend as K 
import tensorflow as tf 

def iou_metrics(Y_true, Y_pred, defaults): 
	"""
		Input: 
			- Y_true, Y_pred: tensors of shape (batch_size, n_defaults, 1 + numClasses + 4)

		Output: 
			- iou_metric: 
				The total overlapping area for positive coords_pred between Y_true and Y_pred
	"""
	# Create a mask for positive boxes
	pos_mask = tf.cast(tf.reduce_max(Y_true[:, :, 1:-4], axis=-1), 'float32')

	defaults = K.cast(tf.constant(defaults), 'float32')

	xy_pred = Y_pred[:, :, -4:-2]*defaults[:, -2:] + defaults[:, :2]
	wh_pred = tf.exp(Y_pred[:, :, -2:])*defaults[:, -2:]

	xy_true = Y_true[:, :, -4:-2]*defaults[:, -2:] + defaults[:, :2]
	wh_true = tf.exp(Y_true[:, :, -2:])*defaults[:, -2:]

	x1, y1 = tf.split(xy_pred, 2, axis=2)
	w1, h1 = tf.split(wh_pred, 2, axis=2)
	x2, y2= tf.split(xy_true, 2, axis=2)
	w2, h2 = tf.split(wh_true, 2, axis=2)

	x12 = x1 + w1
	x22 = x2 + w2
	y12 = y1 + h1
	y22 = y2 + h2

	topleft_x = tf.maximum(x1,x2)
	topleft_y = tf.maximum(y1,y2)

	botright_x = tf.minimum(x12,x22)
	botright_y = tf.minimum(y12,y22)

	intersect = (botright_x - topleft_x)*(botright_y - topleft_y)

	# Calculate areas of every coords_true coords_pred and ground-truth coords_pred
	area_pred = w1*h1
	area_truth = w2*h2

	# Union of area
	union = area_pred + area_truth - intersect

	# Avoid division by 0
	union = tf.maximum(union, 1e-18)

	iou_matrix = tf.maximum(intersect/union, 0)

	# Only keep the positive boxes
	iou_matrix = iou_matrix*pos_mask

	iou_total = K.sum(iou_matrix)/K.sum(pos_mask)

	return iou_total

