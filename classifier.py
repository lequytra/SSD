import numpy as np 
import tensorflow as tf 
import math 
from generateData import generateRandomImages, generateLabels

from matplotlib import pyplot as plt
from mxnet import ndarray
from keras.models import Model
from keras.utils import to_categorical, plot_model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input, BatchNormalization

def Classifier(input_shape, numClasses): 
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

	# FC replaced by Conv
	x = Conv2D(4096, kernel_size=7, padding='valid',
			activation='relu')(x)
	x = Conv2D(4096, kernel_size=1, padding='valid',
			activation='relu')(x)
	result = Conv2D(numClasses, kernel_size=1, padding='valid',
			activation='softmax')(x)

	model = Model(input=input_img, output=result, name="VVG-16")

	return model



classifier = Classifier((224,224,3), 10)

imgs_train = generateRandomImages((600,224,224,3))
labels_train = generateLabels(600, numClass=9)

labels_train = to_categorical(labels_train)

test_imgs = generateRandomImages((100, 224, 224, 3))
test_labels = generateLabels(100, 10)
test_labels = to_categorical(test_labels)
classifier.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
# classifier.summary()
classifier.fit(imgs_train, labels_train, epochs=3, validation_data=(test_imgs, test_labels))
