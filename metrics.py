import keras.backend as K 
import tensorflow as tf 
from tensorflow.math import exp, multiply, add
from tensorflow import maximum, minimum

def iou_metrics(Y_true, Y_pred, defaults): 
	"""
		Input: 
			- Y_true, Y_pred: tensors of shape (batch_size, n_defaults, 1 + numClasses + 4)

		Output: 
			- iou_metric: 
				The total overlapping area for positive coords_pred between Y_true and Y_pred
	"""
	# Create a mask for positive boxes
	pos_mask = tf.cast(tf.reduce_max(Y_true[:, :, 1:-4], axis=-1), 'float64')
	pos_mask = tf.expand_dims(pos_mask, axis=-1)

	Y_true = tf.cast(Y_true, 'float64')
	Y_pred = tf.cast(Y_pred, 'float64')

	defaults = tf.cast(tf.constant(defaults), 'float64')

	xy_pred = add(multiply(Y_pred[:, :, -4:-2], defaults[:, -2:]), defaults[:, :2])
	wh_pred = multiply(exp(Y_pred[:, :, -2:]), defaults[:, -2:])

	xy_true = add(multiply(Y_true[:, :, -4:-2], defaults[:, -2:]), defaults[:, :2])
	wh_true = multiply(exp(Y_true[:, :, -2:]), defaults[:, -2:])

	x1, y1 = tf.split(xy_pred, 2, axis=2)
	w1, h1 = tf.split(wh_pred, 2, axis=2)
	x2, y2= tf.split(xy_true, 2, axis=2)
	w2, h2 = tf.split(wh_true, 2, axis=2)

	x12 = add(x1, w1)
	x22 = add(x2, w2)
	y12 = add(y1, h1)
	y22 = add(y2, h2)

	topleft_x = maximum(x1,x2)
	topleft_y = maximum(y1,y2)

	botright_x = minimum(x12,x22)
	botright_y = minimum(y12,y22)

	intersect = multiply(maximum(botright_x - topleft_x, tf.cast(0, 'float64')), maximum(botright_y - topleft_y, tf.cast(0, 'float64')))

	# Calculate areas of every coords_true coords_pred and ground-truth coords_pred
	area_pred = multiply(w1, h1)
	area_truth = multiply(w2, h2)

	# Union of area
	union = add(add(area_pred, area_truth), multiply(tf.cast(-1, 'float64'), intersect))

	# Avoid division by 0
	union = maximum(union, 1e-8)

	iou_matrix = maximum(tf.math.divide(intersect, union), tf.cast(0, 'float64'))

	# Only keep the positive boxes
	iou_matrix = multiply(iou_matrix, pos_mask)

	num_pos = K.sum(pos_mask)

	def fn(): 
		return tf.math.divide(K.sum(iou_matrix), tf.cast(num_pos, 'float64'))

	def fn0(): 
		return tf.cast(0, 'float64')

	iou_total = tf.cond(tf.equal(num_pos, tf.cast(0, 'float64')), true_fn= lambda : fn0(), false_fn= lambda : fn())

	return iou_total

