import numpy as np 
import tensorflow as tf
from tensorflow.math import argmax, exp
from tensorflow.image import non_maximal_suppression 
import keras.backend as K 

class Decoder(): 
	def __init__(self, 
					predictions, 
					defaults, 
					numClasses=10, 
					iou_thres=0.45, 
					score_thres=0.01, 
					top_k=200):
		"""
			Input: 
				- predictions: the predicted labels and coordinates
							prediction has the form (1 + numClasses + 4)
							The number of predictions must be equal to the number
							of default boxes generated. 
				- defaults : the default boxes for each prediction
							This is to calculate the absolute coordinates
				- numClasses: the number of classes trained
		""" 
		self.defaults = defaults
		self.numClasses = numClasses + 1
		self.background_id = 0
		self.labels = predictions[:, :numClasses + 1]
		self.bboxes = predictions[:, -4:]
		self.iou_thres = iou_thres
		self.score_thres = score_thres
		self.top_k = top_k
		self.decoded = decode_coords()


	def decode_coords(): 
		"""
			Output: 
				- decoded_predictions: 
					decode the prediction into (1 + numClasses + 4 coordinates)
				Coordinates are converted to (x1, y1, x2, y2)

		"""

		n_default = self.defaults.shape[0]

		self.decoded = tf.constant(np.empty(shape=(0, 1 + numClasses + 4)))

		for i in range(n_default): 
			curr_pred = self.bboxes[i]
			curr_db = self.defaults[i]

			xy_abs = curr_pred[:2]*curr_db[2:] + curr_db[:2]
			wh_abs = exp(curr_pred[2:])*curr_db[2:]

			xy2_abs = xy_abs + wh_abs
			
			abs_coords = tf.concat(xy_abs, xy2_abs, axis=0)
			abs_coords = tf.expand_dims(axis=0)

			curr_label = self.labels[i]

			curr_label = tf.expand_dims(axis=0)

			decode_y = tf.concat(curr_label, abs_coords, axis=1)

			self.decoded = tf.concat(self.decoded, decode_y)

		# Get rid of background class score
		self.decoded = self.decoded[:, 1:]

		assert self.decoded.shape == (n_default, numClasses + 4)
		
		return self.decoded 

	def nsm(): 
		max_scores = tf.max(self.labels, axis=1)

		nms_boxes_idx = non_maximal_suppression(boxes=self.decoded[:, -4:], 
											scores=max_scores,
											max_output_size=self.top_k, 
											iou_thres=self.iou_thres, 
											score_thres=self.score_thres)
		return nms_boxes_idx


	def prediction_out(): 

		# Get the class_id with the highest scores
		pred_labels = argmax(self.labels[:, self.background_id + 1:], axis=1)
		selected_boxes_idx = nsm()

		final_pred = tf.concat(pred_labels[selected_boxes_idx], self.decoded[selected_boxes_idx])

		return final_pred
			



