import numpy as np 
import tensorflow as tf 

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

	# Layer 4
	x = Conv2D(512, kernel_size=3, padding='same',
			activation='relu')(x)
	x = Conv2D(512, kernel_size=3, padding='same',
			activation='relu')(x)
	x = Conv2D(512, kernel_size=3, padding='same',
			activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x) 

	# Layer 5 
	x = Conv2D(512, kernel_size=3, padding='same',
			activation='relu')(x)
	x = Conv2D(512, kernel_size=3, padding='same',
			activation='relu')(x)
	x = Conv2D(512, kernel_size=3, padding='same',
			activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x) 

	# Fully-connected layer

	x = Conv2D(4096, kernel_size=7, padding='valid', 
		activation='relu')(x)
	x = Conv2D(4096, kernel_size=1, padding='same', 
		activation='relu')(x)
	x = Conv2D(numClasses, kernel_size=1, padding='same', 
		activation='softmax')(x)
	model = Model(input=input_img, output=x, name="SSD")

	return model 

VGG = build_vgg((224, 224, 3), 10)
VGG.compile(optimizer='sgd', loss='mean_squared_error', 
		metrics=['accuracy'])
VGG.summary()