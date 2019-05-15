import numpy as np 
import tensorflow as tf
from nms import score_suppress, nms, delete_background, top_k, iou
from box_utils import generate_default_boxes

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
		self.predictions = predictions
		self.labels = predictions[:, :-4]
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

		coords = self.bboxes[:, -4:]
		d_coords = self.defaults[:, -4:]

		self.bboxes[:, -4:-2] = coords[:, :-2]*d_coords[:, -2:] + d_coords[:, :-2]
		self.bboxes[:, -2:] = np.exp(coords[:, -2:])*d_coords[:, -2:]
		self.bboxes[:, -2:] = self.bboxes[:, -4:-2] + self.bboxes[:, -2:]


		self.decoded = np.append(self.labels, self.bboxes, axis=1)
		# print("Decoded shape: {}".format(self.decoded.shape))
		
		return self.decoded 


	def prediction_out(self): 

		pred = score_suppress(Y_pred=self.decoded, 
							  numClasses=self.numClasses, 
							  score_thres=self.score_thres)

		# Delete all background boxes
		pred = delete_background(Y_pred=pred, numClasses=self.numClasses)

		# Suppress all boxes
		pred = nms(Y_pred=pred, numClasses=self.numClasses, nms_thres=self.nms_thres)

		# Remove the background column
		pred = pred[:, 1:]

		# Get the class_id with the highest scores
		pred_labels = np.argmax(pred[:, :-4], axis=1)

		# Get the highest scores
		pred_scores = np.amax(pred[:, :-4], axis=1)
		# Cast to float for compatibility
		pred_labels.astype(np.float64)

		pred_labels = np.expand_dims(pred_labels, axis=1)

		pred_scores = np.expand_dims(pred_scores, axis=1)

		# Concat the class id with the box coordinates
		final_pred = np.append(pred_scores, pred[:, -4:], axis=1)
		final_pred = np.append(pred_labels, final_pred, axis=1)
		# Take top k boxes
		pred = top_k(Y_pred=pred, top_k=self.top_k)

		return final_pred

def decode_output(predictions, 
				  numClasses=10,
				  n_layers=6, 
				  min_scale=0.2, 
				  max_scale=0.9, 
				  nms_thres=0.45, 
				  score_thres=0.01, 
				  map_size=[38, 19, 10, 5, 3, 1],
				  top_k=200, 
				  aspect_ratios=[0.5, 1, 2]):

	defaults = generate_default_boxes(n_layers=n_layers, 
										min_scale=min_scale,
										max_scale=max_scale,
										map_size=map_size, 
										aspect_ratios=[0.5, 1, 2])

	decoder = Decoder(predictions=predictions, 
						defaults=defaults, 
						numClasses=numClasses, 
						nms_thres=np.float32(0), 
						score_thres=np.float32(0), 
						top_k=top_k)

	decoded = decoder.prediction_out()

	return decoded

def batch_decoder(predictions, 
				  numClasses=10,
				  n_layers=6, 
				  min_scale=0.2, 
				  max_scale=0.9, 
				  nms_thres=0.45, 
				  score_thres=0.01, 
				  top_k=200, 
				  aspect_ratios=[0.5, 1, 2]): 

	def func(preds):
		return decode_output(predictions=preds, 
							  numClasses=numClasses,
							  n_layers=n_layers, 
							  min_scale=min_scale, 
							  max_scale=max_scale, 
							  nms_thres=nms_thres, 
							  score_thres=score_thres, 
							  top_k=top_k, 
							  aspect_ratios=aspect_ratios)

	n_pred = predictions.shape[0]

	batch_decoder = []

	for i in range(n_pred): 
		decoded = func(predictions[i])
		batch_decoder.append(decoded)

	return batch_decoder


def main(): 
	input_shape=(300, 300, 3)
	numClasses = 10
	iou_thres=0.5 # for default and gt matching
	nms_thres=0.45 # IoU threshold for non-maximal suppression
	score_thres=0.01 # threshold for classification scores
	top_k=5 # the maximum number of predictions kept per image
	min_scale=0.2 # the smallest scale of the feature map
	max_scale=0.9 # the largest scale of the feature map
	aspect_ratios=[0.5, 1, 2] # aspect ratios of the default boxes to be generated
	n_predictions=6 # the number of prediction blocks
	prediction_size=[38, 19, 10, 5, 3, 1] # sizes of feature maps at each level

	Y = np.random.rand(1000, numClasses + 4)

	default = generate_default_boxes(n_layers=n_predictions, 
									min_scale=min_scale, 
									max_scale=max_scale, 
									map_size=prediction_size,
									aspect_ratios=aspect_ratios)


	n_default = default.shape[0]

	predictions = np.random.rand(30, n_default, numClasses + 5)

	decoder = batch_decoder(predictions=predictions, 
							  numClasses=10,
							  n_layers=6, 
							  min_scale=0.2, 
							  max_scale=0.9, 
							  nms_thres=0.45, 
							  score_thres=0.01, 
							  top_k=200, 
							  aspect_ratios=[0.5, 1, 2])

	

	return decoder

if __name__ == '__main__':
	

	results = main()
	print(len(results))

	for i in results: 
		print(i)
		print(i.shape)




