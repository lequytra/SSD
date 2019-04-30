import numpy as np 
import tensorflow as tf 
import math 
from matplotlib import pyplot as plt

from keras.models import Model
from keras.utils import to_categorical, plot_model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input, BatchNormalization, Reshape, Concatenate, Activation
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections

def build_vgg(input_shape, 
			numClasses, 
			mode='training',
			min_scale=0.1, 
			max_scale=0.9,
			aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],
			iou_thres=0.5, 
			top_k=150):

	"""
		Input: 
			- input_shape: 				a tuple specify the image shape (height, width, channels)
			- numClasses: 				the number of classes to be trained on
			- mode: 					a str specifying the mode the model is used for 
			- min_scale: 				The smallest scale of the model's feature map. 
			- max_scale: 				The largest scale of the model's feature map. 
			- aspect_ratios_per_layer:  a list of aspect ratios for the anchor boxes to be generated 
										for each layer
			- iou_thres: 				the cut-off threshold for matching default boxes and ground-truth boxes
			- top_k: 					determine the number of highest predicted outputs to be kept after 
										non-maximal suppression

		Output: 
			- model: 					The Keras SSD model
	"""

	######################################################
	# Compute the parameters for the anchor box generator
	######################################################

	# Setting the configuration for model

	numClasses = numClasses + 1 # Adding a background class
	n_prediction_layer = 6 # The number of prediction blocks the model has
	steps = [None] * n_predictor_layers # Steps is currently not supported
	offsets = [None] * n_predictor_layers # Offsets are currently not supported


	# Calculate the number of boxes for each feature block
	n_boxes = []
	for layer in aspect_ratios_per_layer: 
		n_boxes.append(len(layer) + 1) # An additional aspect ratio of 1 is automatically added

	input_img = Input(shape=input_shape) 

	# Layer 1 
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
	x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x) 

	# Layer 3
	x = Conv2D(256, kernel_size=3, padding='same',
			activation='relu')(x)
	x = Conv2D(256, kernel_size=3, padding='same',
			activation='relu')(x)
	x = Conv2D(256, kernel_size=3, padding='same',
			activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x) 

	# Layer 3b
	x = Conv2D(512, kernel_size=3, padding='same',
			activation='relu')(x)
	x = Conv2D(512, kernel_size=3, padding='same',
			activation='relu')(x)
	x = Conv2D(512, kernel_size=3, padding='same',
			activation='relu')(x)

	featureMap1 = BatchNormalization()(x)
	# Result at scale 1
	class1 = Conv2D(84, kernel_size=3, padding='same')(featureMap1)
	bbox1 = Conv2D(16, kernel_size=3, padding='same')(featureMap1)

	x = Conv2D(512, kernel_size=3, padding='same',
			activation='relu')(x)
	x = Conv2D(512, kernel_size=3, padding='same',
			activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

	# Layer 4
	x = Conv2D(1024, kernel_size=3, padding='same',
			activation='relu')(x)
	# x = Conv2D(1024, kernel_size=1, padding='same',
	# 		activation='relu')(x)
	featureMap2 = Conv2D(1024, kernel_size=1, padding='same',
			activation='relu')(x)
	# Result at scale 2
	class2 = Conv2D(126, kernel_size=3, padding='same')(featureMap2)
	bbox2 = Conv2D(24, kernel_size=3, padding='same')(featureMap2)

	# Layer 5
	x = Conv2D(256, kernel_size=1, padding='same',
			activation='relu')(featureMap2)
	featureMap3 = Conv2D(512, kernel_size=3, strides=2, padding='same',
			activation='relu')(x)

	# Result at scale 3
	class3 = Conv2D(126, kernel_size=3, padding='same')(featureMap3)
	bbox3 = Conv2D(24, kernel_size=3, padding='same')(featureMap3)

	# Layer 6
	x = Conv2D(128, kernel_size=1, padding='valid',
			activation='relu')(featureMap3)
	featureMap4 = Conv2D(256, kernel_size=3, strides=2, padding='same',
			activation='relu')(x)

	# Result at scale 4
	class4 = Conv2D(126, kernel_size=3, padding='same')(featureMap4)
	bbox4 = Conv2D(24, kernel_size=3, padding='same')(featureMap4)

	# Layer 7
	x = Conv2D(128, kernel_size=1, padding='valid',
			activation='relu')(featureMap4)
	featureMap5 = Conv2D(256, kernel_size=3, strides=2, padding='valid',
			activation='relu')(x)

	# Result at scale 5
	class5 = Conv2D(84, kernel_size=3, padding='same')(featureMap5)
	bbox5 = Conv2D(16, kernel_size=3, padding='same')(featureMap5)

	# Layer 7
	x = Conv2D(128, kernel_size=1, padding='valid',
			activation='relu')(featureMap5)
	featureMap6 = Conv2D(256, kernel_size=3, padding='valid',
			activation='relu')(x)

	# Result at scale 4
	class6 = Conv2D(84, kernel_size=3, padding='same')(featureMap6)
	bbox6 = Conv2D(16, kernel_size=3, padding='same')(featureMap6)

	# Generate the anchor boxes
    # Output shape of `anchors`: `(batch, height, width, n_boxes, 8)`
     # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    priorBox1 = AnchorBoxes(input_shape[0], input_shape[1], this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0],
                                              name='priorBox1')(bbox1)
    priorBox2 = AnchorBoxes(input_shape[0], input_shape[1], this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                    		two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1], clip_boxes=clip_boxes,
                                     		name='priorBox2')(bbox2)
    priorBox3 = AnchorBoxes(input_shape[0], input_shape[1], this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2],
                                         name='priorBox3')(bbox3)
   	priorBox4 = AnchorBoxes(input_shape[0], input_shape[1], this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3], 
                                         name='priorBox4')(bbox4)
    priorBox5 = AnchorBoxes(input_shape[0], input_shape[1], this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4], this_offsets=offsets[4], 
                                        name='priorBox5')(bbox5)
    priorBox6 = AnchorBoxes(input_shape[0], input_shape[1], this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5], this_offsets=offsets[5], 
                                        name='priorBox6')(bbox6)

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

    # Reshape the prior boxes to have the shape (batch_size, height*width*n_boxes, 8)
    priorBox1_reshape = Reshape((-1, 8))(priorBox1)
    priorBox2_reshape = Reshape((-1, 8))(priorBox2)
    priorBox3_reshape = Reshape((-1, 8))(priorBox3)
    priorBox4_reshape = Reshape((-1, 8))(priorBox4)
    priorBox5_reshape = Reshape((-1, 8))(priorBox5)
    priorBox6_reshape = Reshape((-1, 8))(priorBox6)

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
    # Concatenate all prior boxes: output shape (batch_size, total_n_boxes, 8)
    prior_concat = Concatenate(axis=1)([priorBox1_reshape, 
    									priorBox2_reshape,
    									priorBox3_reshape,
    									priorBox4_reshape,
    									priorBox5_reshape,
    									priorBox6_reshape])

    # Applying softmax on class predictions across all classes (last axis)
    class_softmax = Activation(axis=-1)(class_concat)

    # Combine all predictions on class scores, bounding boxes and prior boxes 
    # along the last axis. The final output is (batch_size, total_n_boxes, numClass + 1 + 4 + 8)
    predictions = Concatenate(axis=-1)([class_softmax, 
    									bbox_concat, 
    									prior_concat])

    if mode == 'training':
        model = Model(inputs=x, outputs=predictions, name='SSD')

    # If used for inference, the final prediction must be decoded (i.e. convert to absolute coordinates, 
    #							non-maximal suppression, and take top k highest)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               img_height=input_shape[0],
                                               img_width=input_shape[1],
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions, name='SSD')




	return model 

VGG = build_vgg((300, 300, 3), 10)
VGG.compile(optimizer='sgd', loss='mean_squared_error', 
		metrics=['accuracy'])
# (X_train, Y_train), (X_test, Y_test) = build_inputs()
# train(VGG, X_train, Y_train, X_test, Y_test)
# VGG.fit(X_train, Y_train, validation_data=(X_train, Y_train), epochs=3)
VGG.summary()
plot_model(VGG, to_file='model1.png')









