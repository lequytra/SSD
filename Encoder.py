import numpy as np 

class Encoder: 

	def __init__(self, 
				input_shape, 
				numClasses, 
				n_predictions
				min_scale=0.1, 
				max_scale=0.9, 
				aspect_ratio=[1], 
				iou_thres=0.5):
		"""
			Input: 
				- input_shape: the dimension of input image
				- numClasses : The number of classes trained
				- n_predictions: A list of tuples containing the sizes
									of each prediction layer. 
				- min_scale  : The smallest scale of a feature map
				- max_scale  : The largest scale of the feature map
				- aspect_ratio: The aspect ratio of the default boxes used. 
				- iou_thres	 : The threshold to perform matching between default boxes
								ground-truth data

		""" 

		# Convert n_predictions to numpy array
		self.n_predictions = np.array(n_predictions)

		n_layers = n_predictions.shape[0]

		# Generate an array of scales for each prediction level
		self.scales = np.linspace(start=min_scale, end=max_scale, num=n_layers)

		#Generate a list of aspect_ratio for each layer
		self.aspect_ratio = [aspect_ratio]*n_layers

		self.iou_thres = iou_thres

		# Set the number of default boxes
		self.n_boxes = len(aspect_ratio)



		## TODO: generate default boxes

