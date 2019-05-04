import numpy as np 
import tensorflow as tf
from tensorflow.math import argmax
from tensorflow.image import non_maximal_suppression 
import keras.backend as K 

class Decoder(): 
	def __init__(self, 
					predictions, 
					defaults, 
					numClasses=10):
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
		self.predictions = predictions
		self.defaults = defaults
		self.numClasses = numClasses + 1
		self.background_id = 0
		self.labels = predictions[:, :numClasses + 1]
		self.bboxes = predictions[:, -4:]
		self.decoded = decode()


	def decode(): 
		"""
			Output: 
				- decoded_predictions: 
					decode the prediction into (1 + numClasses + 4 coordinates)

		"""
		n_default = self.defaults.shape[0]

		for i in range(n_default): 
			curr_pred = self.bboxes[i]
			curr_db = self.defaults[i]

			xy_abs = K.sum(curr_pred[:2], -1*curr_db[:2])
			wh_abs = 

