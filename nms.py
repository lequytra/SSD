import numpy as np 


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
	print(labels.shape)

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
	print("Y_pred shape: {}".format(Y_pred.shape))
	background_id = 0
	background = Y_pred[:, 0]
	print("Background shape: {}".format(background.shape))

	scores = Y_pred[:, 1:-4]

	coords = Y_pred[:, -4:]

	# Zip corresponding information for sorting 
	boxes_zip = zip(background,scores, coords)

	boxes = [(background, scores, coords) for background, scores, coords in boxes_zip]

	for c in range(numClasses):

		if(c == background_id): 
			continue

		else: 
			# Descendingly sort the boxes based on the scores of the current class
			boxes.sort(key= lambda curr: curr[1][c], reverse=True)

			for i in range(len(boxes) - 1): 
				# Get the current box
				box = boxes[i]
				# If the current box is a background class or 
				# have confidence score of 0

				curr_background, curr_scores, curr_coords = box
				if curr_background == 1 or curr_scores[c] == 0: 
					continue
				else: 
					remaining_boxes = boxes[i + 1:]

					for b in remaining_boxes: 
						_, _, coord = b
						coord = np.expand_dims(coord, axis=0)

						# Expand the shape of current coords to (1, 4)
						curr_coords = np.reshape(curr_coords, (-1, 4))

						iou_scores = iou(curr_coords, coord)

						# for the remaining boxes, suppress all that have high overlapping area
						b[background_id][0] = 1 if iou_scores > nms_thres else b[background_id][0]
						

	# Unzip the boxes variable
	boxes = zip(*boxes)

	Y_suppressed = np.empty(shape=(n_boxes, 0))

	# Append the elements to Y_suppress
	for i in boxes: 
		Y_suppressed = np.append(Y_suppressed, i, axis=1)

	print(Y_suppressed)
	print(type(Y_suppressed))

	return Y_suppressed

def delete_background(Y_pred, numClasses): 
	"""
		A method to delete all background box predictions
	"""
	background = Y_pred[:, 0]

	result = Y_pred[background != 1]

	return result

def top_k(Y_pred, top_k=200): 
	"""
		Return only the top k highest boxes. Boxes should not contain background class
	"""
	n_pred = Y_pred.shape[0]

	if n_pred <= top_k: 
		return Y_pred
	# Find the highest score for each box
	max_scores = np.amax(Y_pred[:, :-4], axis=1)

	scores = Y_pred[:, :-4]
	coords = Y_pred[:, -4:]

	boxes_zip = zip(max_scores, scores, coords)

	boxes = [(max_scores, scores, coords) for max_scores, scores, coords in boxes_zip]

	boxes.sort(key= lambda curr: curr[0], reverse=True)
	result = np.empty(shape=(top_k, 0))

	#Take only the highest k boxes: 
	boxes = boxes[:top_k]
	# Unzip boxes
	boxes = zip(*boxes)

	for i in boxes: 
		result = np.append(result, i, axis=1)

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

	intersect = (botright_x - topleft_x)*(botright_y - topleft_y)
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

