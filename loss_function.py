import tensorflow as tf
import keras
import numpy as np
from keras import backend as K
from keras import losses

def smoothL1(loc_true, loc_pred):
	'''
	https://arxiv.org/pdf/1504.08083.pdf
	Compute smooth L1 loss for location loss
	Input: has been reshaped
		loc_true = (batch_size, n_boxes, 4)
		loc_pred = (batch_size, n_boxes, 4)

	Output
		smoothL1 = (batch_size, n_boxes) 
	'''
	threshold = 1
	x = K.abs(loc_pred - loc_true)
	smooth_x = K.switch(x < threshold, 0.5 * x ** 2, x - 0.5)

	return K.sum(smooth_x, -1) # Sum all slices, across 4 offsets


def logSoftmax(con_true, con_pred):
	'''
	Compute softmax log loss for location loss
	The loss function reaches infinity when input 
	is 0, and reaches 0 when input is 1.
	Input: has been reshaped
		con_true = (batch_size, n_boxes, num_classes)
		con_pred = (batch_size, n_boxes, num_classes)

	Output
		log_loss = (batch_size, n_boxes)
	'''

	# Remove zero input when compute log
	con_pred = tf.math.maximum(con_pred, 0)

	# Compute log softmax loss
	term = con_true * K.log(con_pred)
	log_loss = -K.sum(term, -1)  # Sum all slices, across all classes

	return log_loss

# https://github.com/pierluigiferrari/ssd_keras/blob/master/keras_loss_function/keras_ssd_loss.py
def hard_neg_mining(neg_con_loss_all, num_neg_compute, confidence_loss, 
	batch_size, n_boxes):
	'''
	Compute negative loss for only negative boxes with highest confidence score
	'''
	neg_con_loss_all_1D = K.reshape(neg_con_loss_all, [-1]) # Shape: (batch_size * n_boxes)
	
	values, indices = tf.nn.top_k(neg_con_loss_all_1D,
	            				num_neg_compute,
	                            sorted=False) 
	
	# Create negative mask
	negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
	            				updates=tf.ones_like(indices, dtype=tf.int32),
								shape=tf.shape(neg_con_loss_all_1D)) 

	negatives_keep = K.cast(K.reshape(negatives_keep, [batch_size, n_boxes]),
							'float32') # Shape: (batch_size, n_boxes)

	neg_class_loss = K.sum(confidence_loss * negatives_keep, -1) # Sum across boxes
	
	return neg_class_loss

def loss_function(y_true, y_pred):
	'''
	Weighted sum between location loss and confidence loss
	https://arxiv.org/pdf/1512.02325.pdf
	Input:
		y_true = (batch_size, n_boxes, 1 + num_classes + 4)
		y_pred = (batch_size, n_boxes, 1 + num_classes + 4)
		4: number of offsets including x, y, w, h
		n_boxes: total number of default boxes in each image
	Output:
		A scalar indicating overall loss for location and confidence loss
	'''
	alpha = 1.0
	neg_pos_ratio = 3
	batch_size = K.shape(y_pred)[0]
	n_boxes = K.shape(y_pred)[1]

	if n_boxes == 0:
		total_loss = 0
	else:
		# 1. Calculate location and confidence loss for all boxes
		location_loss = tf.to_float(smoothL1(y_true[:, :, -4:], y_pred[:, :, -4:]))
		confidence_loss = tf.to_float(logSoftmax(y_true[:, :, :-4], y_pred[:, :, :-4]))

		# 2. Calculate confidence and location loss for positive classes
			# Get the maximum score across positive classes, i.e logical matrix 
		pos_mask = tf.to_float(tf.reduce_max(y_true[:, :, 1:-4], axis=-1))

			# Calculate the positive confidence loss
		pos_con_loss = K.sum(confidence_loss * pos_mask, -1) # Sum across boxes

			# Calculate the positive location loss
		pos_loc_loss = K.sum(location_loss * pos_mask, -1) # Sum across boxes

		# 3. Calculate the confidence loss for negative class
		'''
		Many default boxes are categorized as background, which 
		will make the confidence loss imbalance. This is the reason
		for hard negative mining
		'''
			# Create negative class mask
		neg_mask = y_true[:, :, 0] # 0: background class
	
			# Calculate the negative confidence loss
		neg_con_loss_all = confidence_loss * neg_mask

			# Hard negative mining
		num_pos = K.sum(pos_mask)
		num_hard_neg = neg_pos_ratio * K.cast(num_pos, 'int32')

			# Get the total number of negative default boxes
		num_neg_loss = tf.count_nonzero(neg_con_loss_all, dtype=tf.int32)

			# Get the number of negative default boxes 3:1 ratio
		num_neg_compute = K.minimum(num_hard_neg, num_neg_loss)

			# Calculate negative loss
			# If there are no negative boxes, negative loss is 0
		neg_con_loss = K.switch(K.equal(num_neg_loss, tf.constant(0)), 
        	tf.zeros([batch_size]),
        	hard_neg_mining(neg_con_loss_all, 
        		num_neg_compute, confidence_loss, batch_size, n_boxes))

		# 4. Calculate total confidence loss
		con_loss = pos_con_loss + neg_con_loss

		# 5. Calculate the total loss
		num_pos = K.cast(num_pos, 'float32')
		# In case there are no positive boxes
		total_loss = (con_loss + alpha * pos_loc_loss) / tf.math.maximum(K.cast(1.0, 'float32'), num_pos)

	return total_loss
	