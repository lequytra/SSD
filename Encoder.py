import numpy as np 
from math import sqrt
import tensorflow as tf 
import keras.backend as K 
from sklearn.model_selection import train_test_split
import itertools as it 

from box_utils import IoU, generate_default_boxes



class Encoder(): 
	def __init__(self, 
				y_truth, 
				default, 
				numClasses=10,
				input_shape=(300,300,3),
				iou_thres=0.5): 
		""" 
			Input: 
				- input_shape: the shape of the input image
				- numClasses: the number of classes that will be trained
				- y_truth: the ground-truth labels of the image ([])
				- iou_thres: The threshold for matching default to ground-truth
				- min_scale: the smallest feature map scale
				- max_scale: the largest feature map scale
				- aspect_ratio: a list of aspect ratios for each default boxes
				- n_predictions: number of prediction layers
				- prediction_size: a list of sizes for the predictions
									The number of element must be equal to n_predictions
		"""

		self.numClasses = numClasses 
		self.iou_thres = iou_thres
		self.default = default
		self.background_id = 0
		self.labels = y_truth[:, :-4]
		self.boxes = y_truth[:, -4:]
		self.iou_matrix = IoU(self.default, self.boxes)
		self.matches = self.multi_matching()


	def encode_format(self):
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

	def max_bipartite_matching(self): 
		"""
			Perform the maximum bipartite matching algorithm that matches the 
			ground-truth box with a default with highest Jaccard index

			Output: 
				- Return a 1D array of length n_boxes that contains the index of 
					the matched default boxes for each ground-truth box.  

		"""

		n_boxes = self.boxes.shape[0]
		weight = self.iou_matrix
		matched = np.zeros((n_boxes,))

		for i in range(n_boxes): 
			max_index = np.unravel_index(np.argmax(weight, axis=None), self.iou_matrix.shape)
			gt_coord = max_index[1]
			db_coord = max_index[0]
			matched[gt_coord] = db_coord
			weight[:, gt_coord] = 0
			weight[db_coord, :] = 0
			self.iou_matrix[db_coord, gt_coord] = 1

		return matched


	def multi_matching(self): 
		"""
			Match the default boxes to any ground-truth boxes with 
			iou >= iou_thres
			If none, set to -1

			Output: 
				- matches: The index of the ground-truth box matched
								with each default box (n_default,)
		"""
		matched = self.max_bipartite_matching()
		highest_box = np.max(self.iou_matrix, axis=1)

		assert highest_box.shape[0] == self.default.shape[0]

		self.matches = np.argmax(self.iou_matrix, axis=1)

		# Set all the matched pair with iou < thres to -1
		self.matches[highest_box < self.iou_thres] = -1

		assert self.matches.shape[0] == self.default.shape[0]


		return self.matches


	def get_encoded_data(self):

		n_default = self.default.shape[0]

		# Generate a template for the encoded labels (#default, 1 + numClasses + 4)
		encoded = np.empty(shape=(0, self.numClasses + 4))

		for i in range(n_default):
			
			matched_gt =  self.matches[i]
			# If the default box is not matched with any ground-truth
			if matched_gt == -1: 
				encoded_y = np.zeros(shape=(1, self.numClasses + 4))
				encoded = np.append(encoded, encoded_y, axis=0)

			else: 
				curr_default = self.default[i]

				match = self.boxes[matched_gt] # (x, y, w, h) normalized

				# Calculate the offset of the matched ground-truth to the default box
				xy_offset = (match[:2] - curr_default[:2])/curr_default[2:]
				wh_offset = np.log(match[2:]/curr_default[2:])

				assert xy_offset.shape == (2,)

				label = self.labels[matched_gt, :]

				# Append to offset (x, y, w, h)
				encoded_y = np.append(label, [xy_offset, wh_offset])
				encoded_y = np.expand_dims(encoded_y, axis=0)

				assert encoded_y.shape == (1, self.numClasses + 4)

				encoded = np.append(encoded, encoded_y, axis=0)

		# The default that is not matched with any ground-truth is considered
		# the background class
		background_class = self.matches < 0
		background_class = np.expand_dims(background_class, axis=1)


		# Append background class to produce the final encoded labels
		encoded = np.append(background_class, encoded, axis=1)

		assert encoded.shape == (n_default, 1 + self.numClasses + 4)

		return encoded

def encode_batch(y_truth, 
				default, 
				numClasses=10,
				input_shape=(300,300,3),
				iou_thres=0.5): 
	
	func = lambda Y : Encoder(y_truth=Y, 
							default=default,
			                numClasses=numClasses, 
			                input_shape=input_shape,
			                iou_thres=iou_thres).get_encoded_data()

	encoded_all = [func(Y) for Y in y_truth]

	encoded_all = np.array(encoded_all)

	return encoded_all




def main(Y): 
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

	default = generate_default_boxes(n_layers=n_predictions, 
									min_scale=min_scale, 
									max_scale=max_scale, 
									map_size=prediction_size,
									aspect_ratios=aspect_ratios)
	# encode = Encoder(y_truth=Y, 
	# 				default=default,
	#                 numClasses=numClasses, 
	#                 iou_thres=iou_thres,
	#                 aspect_ratios=aspect_ratios)

	# Y = encode.get_encoded_data()

	Y = encode_batch(y_truth=Y, 
					default=default,
	                numClasses=numClasses, 
	                input_shape=(300,300,3),
	                iou_thres=iou_thres)

	return Y

# if __name__ == '__main__':
	

	# X = np.random.rand(100, 300, 300, 3)
	# Y = np.random.rand(100, 3, 14)

	# Y_train = main(Y)
	
	# print(Y_train.shape)

	# print(type(Y_train))

