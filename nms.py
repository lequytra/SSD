import numpy as np 
from box_utils import IoU

def score_suppress(Y_pred, 
					numClasses=10,
					score_thres=0.01): 
	"""
		Eliminate all box of background classes and classes with highest
			score smaller than threshold
	"""
	# Get all label predictions
	labels = Y_pred[:, :-4]
	# Find the highest score in each class
	max_label = np.amax(labels, axis=1)


	# For each box that has highest scores lower than threshold, 
	# Set the background to 1

	Y_pred[max_label < score_thres, 0] = 1

	# find the class with the highest scores
	highest_class = np.argmax(labels, axis=1)
	# If it is most likely to be background, set background to 1
	Y_pred[highest_class == 0, 0] = 1

	return Y_pred

def nms(Y_pred,
		numClasses=10, 
		nms_thres=0.45): 
	"""
		Input: 
			- Y_pred 	: a numpy array of all predictions
				Must be in the format (n_default, 1 + numClasses + 4)
			- numClasses: the number of boxes predicted
			- nms_thres : threshold for non-maximum suppression

		Output: 
			- Y_suppressed: a tensor of predictions that satisfy
							shape (<= k, 1 + numClasses + 4)

	"""
	n_boxes = Y_pred.shape[0]
	
	background_id = 0
	background = Y_pred[:, 0]

	scores = Y_pred[:, 1:-4]

	coords = Y_pred[:, -4:]

	# Turn into lists of boxes
	boxes = [Y_pred[i, :] for i in range(n_boxes)]

	picked = np.empty(shape=(0, 1 + numClasses + 4))

	for c in range(numClasses):

		if(c == background_id): 
			continue

		else: 
			# Descendingly sort the boxes based on the scores of the current class
			boxes.sort(key= lambda curr: curr[c], reverse=True)
			# print("Shape of boxes: {}".format(len(boxes)))

			remaining = np.stack(boxes, axis=0)

			while(remaining.shape[0] > 0): 
				# Get the highest box
				curr_coords = remaining[0, -4:]
				# print("Curr shape: {}".format(remaining[0].shape))
				# Add the current highest to the set 
				picked = np.append(picked, np.expand_dims(remaining[0], axis=0), axis=0)

				curr_coords = np.expand_dims(curr_coords, axis=0)

				rest_coords = remaining[:, -4:]
				# print("Shape of curr coords and rest: {}, {}".format(curr_coords.shape, rest_coords.shape))
				# Calculate the IoU with rest: 

				iou_scores = IoU(rest_coords, curr_coords)
				# print("Shape of iou_scores: {}".format(iou_scores.shape))
				# Make it a 1D array
				iou_scores = np.squeeze(iou_scores, axis=1)

				remaining = remaining[iou_scores <= nms_thres]

				picked = np.append(picked, remaining, axis=0)

	# Get only unique boxes
	picked = np.unique(picked, axis=0)
	# Append to be a 2D np array
	result = np.stack(picked, axis=0)

	return result

	

def delete_background(Y_pred, numClasses): 
	"""
		A method to delete all background box predictions
	"""
	# Get the background class
	background = Y_pred[:, 0]

	# Only get boxes that are not background
	result = Y_pred[background != 1]

	return result

def top_k(Y_pred, top_k=200): 
	"""
		Return only the top k highest boxes. Boxes should not contain background class
	"""
	n_boxes = Y_pred.shape[0]

	if n_boxes <= top_k: 
		return Y_pred
	# Find the highest score for each box
	max_scores = np.amax(Y_pred[:, :-4], axis=1)

	scores = Y_pred[:, :-4]
	coords = Y_pred[:, -4:]

	boxes = [Y_pred[i, :] for i in range(n_boxes)]

	boxes.sort(key= lambda curr: curr[0], reverse=True)

	#Take only the highest k boxes: 
	boxes = boxes[:top_k]

	result = np.stack(boxes, axis=0)

	return result




def iou(box1, box2): 
	"""
		Input: 
			- box1: an array of coordinates of box1 boxes
			- box2   : an array of coordinates of box2round-truth boxes
			coordinates are in format (x_l, y_l, x_r, y_r)
		Output: 
			- iou:  a 2D tensor of shape (n_box1, n_truth), returning the
						Jaccard index of each box2 boxes for
						every ground-truth boxes. box2	"""

	x1, y1, x12, y12 = np.split(box1, 4, axis=1)
	x2, y2, x22, y22 = np.split(box2, 4, axis=1)


	topleft_x = np.maximum(x1,x2)
	topleft_y = np.maximum(y1,y2)

	botright_x = np.minimum(x12,x22)
	botright_y = np.minimum(y12,y22)

	intersect = np.maximum(botright_x - topleft_x, 0)*np.maximum(botright_y - topleft_y, 0)
	# Calculate areas of every box1 boxes and ground-truth boxes
	area = (x12 - x1)*(y12 - y1) + (x2 - x22)*(y2 - y22)

	# Union of area
	union = area - intersect
	# Avoid division by 0
	union = np.maximum(union, 1e-18)

	iou_matrix = np.maximum(intersect/union, 0)

	return iou_matrix


# def call(): 
# 	Y_pred = np.random.rand(11, 15)
# 	s = nms(Y_pred=Y_pred)


# call()

