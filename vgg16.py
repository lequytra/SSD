import numpy as np 
import tensorflow as tf 
import math 
from gluoncv import data, utils
from matplotlib import pyplot as plt
from mxnet import ndarray
from keras.datasets import mnist
from keras.models import Model
from keras.utils import to_categorical, plot_model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input, BatchNormalization

def build_vgg(input_shape, numClasses):

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

	# ------------------------------------------
	# # FC Layer replaced by Conv
	# x = Conv2D(4096, kernel_size=7, padding='valid',
	# 		activation='relu')(x)
	# x = Conv2D(4096, kernel_size=1, padding='valid',
	# 		activation='relu')(x)
	# result = Conv2D(numClasses, kernel_size=1, padding='valid',
	# 		activation='relu')(x)

	# ------------------------------------------

	model = Model(input=input_img, output=[class1, bbox1, class2, bbox2,
		class3, bbox3, class4, bbox4, class5, bbox5, class6, bbox6], name="SSD")

	return model 

VGG = build_vgg((300, 300, 3), 10)
VGG.compile(optimizer='sgd', loss='mean_squared_error', 
		metrics=['accuracy'])
# (X_train, Y_train), (X_test, Y_test) = build_inputs()
# train(VGG, X_train, Y_train, X_test, Y_test)
# VGG.fit(X_train, Y_train, validation_data=(X_train, Y_train), epochs=3)
VGG.summary()
plot_model(VGG, to_file='model1.png')









