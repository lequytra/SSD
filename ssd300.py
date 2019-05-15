import numpy as np 
import tensorflow as tf 
import math 
from matplotlib import pyplot as plt
from box_utils import generate_default_boxes
from loss_function import loss_function
from metrics import iou_metrics
from keras import losses
from keras.models import Model
from keras.utils import to_categorical, plot_model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input, BatchNormalization, Reshape, Concatenate, Softmax
from keras.optimizers import Adam

def build_SSD300(input_shape, 
			numClasses, 
			mode='training',
			min_scale=0.2, 
			max_scale=0.9,
			aspect_ratios=[0.5, 1, 2],
			iou_thres=0.5, 
			nms_thres=0.45,
			score_thres=0.01,
			top_k=150, 
			n_predictions=6):

	"""
		Input: 
			- input_shape: 				a tuple specify the image shape (height, width, channels)
			- numClasses: 				the number of classes to be trained on
			- mode: 					a str specifying the mode the model is used for 
			- min_scale: 				The smallest scale of the model's feature map. 
			- max_scale: 				The largest scale of the model's feature map. 
			- aspect_ratios:  			a list of aspect ratios for the anchor boxes to be generated 
										for each layer
			- iou_thres: 				the cut-off threshold for matching default boxes and ground-truth boxes
			- top_k: 					determine the number of highest predicted outputs to be kept after 
										non-maximal suppression

		Output: 
			- model: 					The Keras SSD model
	"""

	#######################################################
	# Compute the parameters for the anchor box generator #
	#######################################################

	# Setting the configuration for model

	numClasses = numClasses + 1 # Adding a background class
	# The number of default boxes at each location
	n_default = len(aspect_ratios)

	input_img = Input(shape=input_shape) 

	# Calculate the number of neurons for the classifier
	n_classifier = n_default*numClasses
	# Calculate the number of neurons for the regressor
	n_regressor = n_default*4

	######################
	# 	Base Network	 #
	######################

	x = Conv2D(64, kernel_size=3, padding='same',
			activation='relu')(input_img)
	x = Conv2D(64, kernel_size=3, padding='same',
			activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

	# Layer 2
	x = Conv2D(128, kernel_size=3, padding='same',
			activation='relu')(x)
	x = Conv2D(128, kernel_size=3, padding='same',
			activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x) 

	# Layer 3
	x = Conv2D(256, kernel_size=3, padding='same',
			activation='relu')(x)
	x = Conv2D(256, kernel_size=3, padding='same',
			activation='relu')(x)
	x = Conv2D(256, kernel_size=3, padding='same',
			activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x) 

	# Layer 4
	x = Conv2D(512, kernel_size=3, padding='same',
			activation='relu')(x)
	x = Conv2D(512, kernel_size=3, padding='same',
			activation='relu')(x)
	x = Conv2D(512, kernel_size=3, padding='same',
			activation='relu')(x)

	featureMap1 = BatchNormalization()(x)

	######################
	# Prediction Layer 1 #
	######################

	class1 = Conv2D(n_classifier, kernel_size=3, padding='same')(featureMap1)
	bbox1 = Conv2D(n_regressor, kernel_size=3, padding='same')(featureMap1)

	######################
	# 	   Block 2		 #
	######################	

	x = Conv2D(1024, kernel_size=3, padding='same',
			activation='relu')(x)
	x = Conv2D(1014, kernel_size=1, padding='same',
			activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

	featureMap2 = Conv2D(1024, kernel_size=1, padding='same',
			activation='relu')(x)

	######################
	# Prediction Layer 2 #
	######################

	class2 = Conv2D(n_classifier, kernel_size=3, padding='same')(featureMap2)
	bbox2 = Conv2D(n_regressor, kernel_size=3, padding='same')(featureMap2)

	######################
	# 	   Block 3		 #
	######################

	x = Conv2D(256, kernel_size=1, padding='same',
			activation='relu')(featureMap2)
	featureMap3 = Conv2D(512, kernel_size=3, strides=2, padding='same',
			activation='relu')(x)

	######################
	# Prediction Layer 3 #
	######################

	class3 = Conv2D(n_classifier, kernel_size=3, padding='same')(featureMap3)
	bbox3 = Conv2D(n_regressor, kernel_size=3, padding='same')(featureMap3)

	######################
	# 	   Block 4		 #
	######################

	x = Conv2D(128, kernel_size=1, padding='same',
			activation='relu')(featureMap3)
	featureMap4 = Conv2D(256, kernel_size=3, strides=2, padding='same',
			activation='relu')(x)

	######################
	# Prediction Layer 4 #
	######################

	class4 = Conv2D(n_classifier, kernel_size=3, padding='same')(featureMap4)
	bbox4 = Conv2D(n_regressor, kernel_size=3, padding='same')(featureMap4)

	######################
	# 	   Block 5		 #
	######################

	x = Conv2D(128, kernel_size=1, padding='same',
			activation='relu')(featureMap4)
	featureMap5 = Conv2D(256, kernel_size=3, strides=2, padding='same',
			activation='relu')(x)

	######################
	# Prediction Layer 5 #
	######################

	class5 = Conv2D(n_classifier, kernel_size=3, padding='same')(featureMap5)
	bbox5 = Conv2D(n_regressor, kernel_size=3, padding='same')(featureMap5)

	######################
	# 	   Block 6		 #
	######################
	x = Conv2D(128, kernel_size=1, padding='valid',
			activation='relu')(featureMap5)
	featureMap6 = Conv2D(256, kernel_size=3, padding='valid',
			activation='relu')(x)

	######################
	# Prediction Layer 6 #
	######################

	class6 = Conv2D(n_classifier, kernel_size=3, padding='same')(featureMap6)
	bbox6 = Conv2D(n_regressor, kernel_size=3, padding='same')(featureMap6)

    # Reshape the classification outputs to have the shape (batch_size, height*width*n_boxes, numClasses)
	class1_reshape = Reshape((-1, numClasses))(class1)
	class2_reshape = Reshape((-1, numClasses))(class2)
	class3_reshape = Reshape((-1, numClasses))(class3)
	class4_reshape = Reshape((-1, numClasses))(class4)
	class5_reshape = Reshape((-1, numClasses))(class5)
	class6_reshape = Reshape((-1, numClasses))(class6)

    # Reshape the bbox outputs to have the shape (batch_size, height*width*n_boxes, 4)
	bbox1_reshape = Reshape((-1, 4))(bbox1)
	bbox2_reshape = Reshape((-1, 4))(bbox2)
	bbox3_reshape = Reshape((-1, 4))(bbox3)
	bbox4_reshape = Reshape((-1, 4))(bbox4)
	bbox5_reshape = Reshape((-1, 4))(bbox5)
	bbox6_reshape = Reshape((-1, 4))(bbox6)

    # Concatenate classes: output shape (batch_size, total_n_boxes, numClasses + 1)

	class_concat = Concatenate(axis=1)([class1_reshape, 
										class2_reshape,
										class3_reshape,
										class4_reshape,
										class5_reshape,
										class6_reshape])

    # Concatenate all bounding box predictions: output shape (batch_size, total_n_boxes, 4)
	bbox_concat = Concatenate(axis=1)([bbox1_reshape, 
										bbox2_reshape,
										bbox3_reshape,
										bbox4_reshape,
										bbox5_reshape,
										bbox6_reshape])

    # Applying softmax on class predictions across all classes (last axis)
	class_softmax = Softmax(axis=-1)(class_concat)

    # Combine all predictions on class scores, bounding boxes and prior boxes 
    # along the last axis. The final output is (batch_size, total_n_boxes, numClass + 1 + 4 + 8)

    ######################
	#  Final Prediction  #
	######################

	predictions = Concatenate(axis=-1)([class_softmax, 
										bbox_concat])

	if mode == 'training':
	    model = Model(inputs=input_img, outputs=predictions, name='SSD-300')

	# elif mode == 'inference': 
	# 	encoder = Encoder()
	# 	default = encoder.default
	# 	predictions = Decoder(predictions=predictions,
	# 							defaults=default,
	# 							numClasses=numClasses - 1, 
	# 							nms_thres=nms_thres, 
	# 							score_thres=score_thres, 
	# 							top_k=top_k).nsm()
	# 	model = Model(inputs=input_img, outputs=predictions, name='SSD-300')

	return model 

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
prediction_size=[38, 19, 10, 5, 3, 1] # sizes of feature maps at each level

# Generate default boxes: 
default = generate_default_boxes(n_layers=n_predictions, 
                                min_scale=min_scale, 
                                max_scale=max_scale, 
                                map_size=prediction_size, 
                                aspect_ratios=aspect_ratios)

adam = Adam()

def iou(Y_true, Y_pred): 
	return iou_metrics(Y_true, Y_pred, default)

SSD300 = build_SSD300((300, 300, 3), 10)
SSD300.compile(optimizer=adam, loss=loss_function, 
		metrics=['accuracy', iou])
# # (X_train, Y_train), (X_test, Y_test) = build_inputs()
# # train(SSD300, X_train, Y_train, X_test, Y_test)
# # SSD300.fit(X_train, Y_train, validation_data=(X_train, Y_train), epochs=3)
# SSD300.summary()
# plot_model(SSD300, to_file='model1.png')
