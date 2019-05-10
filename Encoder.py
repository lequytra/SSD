import numpy as np 
from math import sqrt
import tensorflow as tf 
import keras.backend as K 
from sklearn.model_selection import train_test_split
import itertools as it 



class Encoder(): 
	def __init__(self, 
				y_truth, 
				numClasses=10,
				input_shape=(300,300,3),
				iou_thres=0.5,
				min_scale=0.2, 
				max_scale=0.9, 
				aspect_ratios=[0.5, 1, 2], 
				n_predictions=6, 
				prediction_size=[38, 19, 10, 5, 3, 1]): 
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

		self.im_height = input_shape[0]
		self.numClasses = numClasses 
		self.y_truth = y_truth
		self.im_width = input_shape[1]
		self.n_layers = n_predictions
		self.aspect_ratios = aspect_ratios
		self.iou_thres = iou_thres
		# Calculate the scale at each prediction layer
		self.scales = np.linspace(start=min_scale, stop=max_scale, num=n_predictions)
		self.map_size = prediction_size
		self.default = self.generate_default_boxes()
		self.background_id = 0
		self.labels = y_truth[:, :-4]
		self.boxes = y_truth[:, -4:]
		self.iou_matrix = self.IoU()
		self.matches = self.multi_matching()



	def generate_default_boxes(self): 

		"""
			Output: 
				- default: a 2D array (#defaults, 4) containing the coordinates [x, y, h, w] 
							of all default boxes relative to the image size. 
		"""
		self.default = np.empty(shape=(0, 4))

		for level in range(self.n_layers):

			scale = self.scales[level] 
			# For each pixel location in the feature map
			for i, j in it.product(range(self.map_size[level]), repeat=2): 
				
				# Calculate the center of each default box
				x = (i + 0.5)/self.map_size[level]
				y = (j + 0.5)/self.map_size[level]

				for ratio in self.aspect_ratios: 
					box = np.empty(shape=(1, 4))

					box[:, 0] = x
					box[:, 1] = y

					# Calculate the width and height
					w = scale*sqrt(ratio)
					h = scale/sqrt(ratio)

					box[:, 2] = w
					box[:, 3] = h

					self.default = np.concatenate((self.default, box), axis=0)
					
		return self.default

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

	def IoU(self): 
		"""

			Output: 
				- iou:  a 2D tensor of shape (n_default, n_truth), returning the
							Jaccard index of each default boxes for
							every ground-truth boxes. 
		"""

		x1, y1, w1, h1 = np.split(self.default, 4, axis=1)
		x2, y2, w2, h2 = np.split(self.boxes, 4, axis=1)

		x12 = x1 + w1
		x22 = x2 + w2
		y12 = y1 + h1
		y22 = y2 + h2

		n_default = self.default.shape[0]
		n_truth = self.y_truth.shape[0]

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

		self.iou_matrix = np.maximum(intersect/union, 0)

		return self.iou_matrix

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
				xy_offset = match[:2] - curr_default[:2]
				wh_offset = match[2:]/curr_default[2:]

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
		print(background_class.shape)
		print(encoded.shape)

		# Append background class to produce the final encoded labels
		encoded = np.append(background_class, encoded, axis=1)

		assert encoded.shape == (n_default, 1 + self.numClasses + 4)

		return encoded

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

	encode = Encoder(y_truth=Y, 
                numClasses=numClasses, 
                iou_thres=iou_thres,
                min_scale=min_scale, 
                max_scale=max_scale, 
                aspect_ratios=aspect_ratios, 
                n_predictions=n_predictions, 
                prediction_size=prediction_size)
	Y = encode.get_encoded_data()
	

	return Y

if __name__ == '__main__':
	

	X = np.random.rand(1000, 300, 300, 3)
	Y = np.random.rand(1000, 14)

	Y_train = main(Y)
	
	print(Y_train.shape)

	print(type(Y_train))

