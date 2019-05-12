import numpy as np 
from box_utils import IoU



def nms(Y_pred,
		numClasses=10, 
		nms_thres=0.45, 
		score_thres=0.01): 
	"""
		Input: 
			- Y_pred 	: a numpy array of all predictions
				Must be in the format (n_default, 1 + numClasses + 4)
			- numClasses: the number of classes predicted
			- nms_thres : threshold for non-maximum suppression
			- score_thres: the threshold for class scores

		Output: 
			- Y_suppressed: a tensor of predictions that satisfy
							shape (<= k, 1 + numClasses + 4)

	"""
	n_boxes = Y_pred.shape[0]
	background_id = 0
	background = Y_pred[:, 0]
	print("background shape: {}".format(background.shape))
	scores = Y_pred[:, 1:numClasses+1]

	max_scores = np.max(scores, axis=1)
	print("max_scores shape: {}".format(max_scores.shape))
	# Suppress all predictions with the highest class score smaller than threshold
	background[max_scores < score_thres] = 1
	background = np.expand_dims(background, axis=1)
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
				print("Curr box shape: {}".format(len(box)))
				curr_background, curr_scores, curr_coords = box
				print("Curr background shape: {}".format(curr_background.shape))
				if curr_background == 1 or curr_scores[c] == 0: 
					continue
				else: 
					print("remaining: {}".format(len(boxes[i+1:])))
					remaining_boxes = boxes[i + 1:]

					remaining_coords = np.empty(shape=(0, 4))
					for b in remaining_boxes: 
						_, _, coord = b
						coord = np.expand_dims(coord, axis=0)
						remaining_coords = np.append(remaining_coords, coord, axis=0)

					print("shape of remaining coords: {}".format(remaining_coords.shape))
					print("remaining coords: {}".format(remaining_coords))

					_, _, curr_coords = box
					# Expand the shape of current coords to (1, 4)
					curr_coords = np.expand_dims(curr_coords, axis=0)
					print("Shape of curr_coords: {}".format(curr_coords.shape))

					iou_scores = IoU(curr_coords, remaining_coords)

					print("Shape of iou_matrix: {}".format(iou_scores.shape))

					# for the remaining boxes, suppress all that have high overlapping area
					boxes[i+2:][background_id][iou_scores > nms_thres] = 1

	# Unzip the boxes variable
	boxes = zip(*boxes)
	boxes = [np.array(element) for element in boxes]

	Y_suppressed = np.empty(shape=(n_boxes, 0))

	# Append the elements to Y_suppress
	for i in boxes: 
		Y_suppressed = np.append(Y_suppressed, i, axis=1)

	print("Initial number of boxes: {}".format(n_boxes))
	print("Shape after suppressed: {}".format(Y_suppressed.shape))

	return Y_suppressed

def call(): 
	Y_pred = np.random.rand(11, 15)
	s = nms(Y_pred=Y_pred)


call()

