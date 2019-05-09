import numpy as np 
from Encoder import Encoder
import tensorflow as tf
from tensorflow.math import argmax, exp, reduce_max
from tensorflow.image import non_max_suppression 
import keras.backend as K 

class Decoder(): 
	def __init__(self, 
					predictions, 
					defaults, 
					numClasses=10, 
					nms_thres=0.45, 
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
		self.numClasses = numClasses
		self.background_id = 0
		self.labels = predictions[:, :numClasses + 1]
		self.bboxes = predictions[:, -4:]
		self.nms_thres = nms_thres
		self.score_thres = score_thres
		self.top_k = top_k
		self.decoded = self.decode_coords()


	def decode_coords(self): 
		"""
			Output: 
				- decoded_predictions: 
					decode the prediction into (1 + numClasses + 4 coordinates)
				Coordinates are converted to (x1, y1, x2, y2)

		"""

		n_default = self.defaults.shape[0]

		self.decoded = tf.constant(np.empty(shape=(0, 1 + self.numClasses + 4)))

		for i in range(n_default): 
			curr_pred = self.bboxes[i]
			curr_db = self.defaults[i]

			xy_abs = curr_pred[:2] + curr_db[:2]
			wh_abs = curr_pred[2:]*curr_db[2:]

			xy2_abs = xy_abs + wh_abs
			
			abs_coords = tf.concat([xy_abs, xy2_abs], axis=0)
			# abs_coords = tf.expand_dims(abs_coords, axis=0)

			curr_label = self.labels[i]

			# curr_label = tf.expand_dims(axis=0)

			decode_y = tf.concat([curr_label, abs_coords], axis=0)

			decode_y = tf.expand_dims(decode_y, axis=0)

			self.decoded = tf.concat([self.decoded, decode_y], axis=0)

		# Get rid of background class score
		self.decoded = self.decoded[:, 1:]

		assert self.decoded.shape == (n_default, self.numClasses + 4)
		
		return self.decoded 

	def nms(self): 
		max_scores = reduce_max(self.labels, axis=1)

		nms_boxes_idx = non_max_suppression(boxes=self.decoded[:, -4:], 
											scores=max_scores,
											max_output_size=self.top_k, 
											iou_threshold=self.nms_thres, 
											score_threshold=self.score_thres)
		return nms_boxes_idx


	def prediction_out(self): 

		# Get the class_id with the highest scores
		pred_labels = argmax(self.labels[:, self.background_id + 1:], axis=1)
		selected_boxes_idx = self.nms()

		final_pred = tf.concat(pred_labels[selected_boxes_idx], self.decoded[selected_boxes_idx])

		return final_pred
			
def main(): 
	input_shape=(300, 300, 3)
	numClasses = 10
	iou_thres=0.5 # for default and gt matching
	nms_thres=0.45 # IoU threshold for non-maximal suppression
	score_thres=0.01 # threshold for classification scores
	top_k=200 # the maximum number of predictions kept per image
	min_scale=0.2 # the smallest scale of the feature map
	max_scale=0.9 # the largest scale of the feature map
	aspect_ratios=[0.5, 1, 2] # aspect ratios of the default boxes to be generated
	n_predictions=6 # the number of prediction blocks
	prediction_size=[37, 18, 10, 5, 3, 1] # sizes of feature maps at each level

	Y = np.random.rand(1000, numClasses + 4)

	encoder = Encoder(y_truth=Y, 
	                numClasses=numClasses, 
	                iou_thres=iou_thres,
	                min_scale=min_scale, 
	                max_scale=max_scale, 
	                aspect_ratios=aspect_ratios, 
	                n_predictions=n_predictions, 
	                prediction_size=prediction_size)
	defaults = encoder.default

	n_default = defaults.shape[0]

	predictions = np.random.rand(n_default, numClasses + 5)

	decoder = Decoder(predictions=predictions, 
					defaults=defaults, 
					numClasses=numClasses, 
					nms_thres=nms_thres, 
					score_thres=score_thres, 
					top_k=top_k)

	return decoder

if __name__ == '__main__':
	decoder = main()

	results = decoder.prediction_out()

	print(result.shape)
	print(type(results))




