import keras
from keras.applications import VGG19

vgg = VGG19(weights=None,classes=25,input_shape=(300,300,3))
vgg.compile(optimizer="adam",loss='categorical_crossentropy')
vgg.summary()