from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, LearningRateScheduler, TensorBoard, RemoteMonitor
from keras import backend as K
from keras.models import load_model
from keras.callbacks import History, ModelCheckpoint
from sklearn.model_selection import train_test_split
from ssd300 import build_SSD300
from parser import Parser
import matplotlib.pyplot as plt
import numpy as np
import os
from metrics import iou_metrics
from Decoder import Decoder
from Encoder import Encoder, encode_batch
from box_utils import IoU, generate_default_boxes
from loss_function import loss_function
from keras import metrics, losses
import h5py
import requests
from time import time


os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Configuration of the model: 
input_shape=(300, 300, 3)
numClasses = 70
iou_thres=0.5 # for default and gt matching
nms_thres=0.45 # IoU threshold for non-maximal suppression
score_thres=0.01 # threshold for classification scores
top_k=200 # the maximum number of predictions kept per image
min_scale=0.2 # the smallest scale of the feature map
max_scale=0.9 # the largest scale of the feature map
aspect_ratios=[0.5, 1, 2] # aspect ratios of the default boxes to be generated
n_predictions=6 # the number of prediction blocks
prediction_size=[38, 19, 10, 5, 3, 1] # sizes of feature maps at each level

print("Parsing Data ... \n")

data_dir = "/Users/tranle/mscoco"
training_data = "val2017"
# Initialize a parser object
parser = Parser(data_dir, training_data, numClasses)

# Load images and annotations for the image
# For now, we load only 10 first classes and images are resize to (300,300,3) 
# for training purposes

X, Y = parser.load_data()

X = np.array(X)
Y = np.array(Y)
print("Shape of parsed images: {}".format(X.shape))
print("Shape of parsed labels: {}".format(Y.shape))
print("Shape of one label: {}\n\n".format(Y[0].shape))

# Generate default boxes: 
default = generate_default_boxes(n_layers=n_predictions, 
                                min_scale=min_scale, 
                                max_scale=max_scale, 
                                map_size=prediction_size, 
                                aspect_ratios=aspect_ratios)


print("Encoding data ... \n")


# Get 2000 images for training
X_train = X
Y_train = Y

# Encode the labels and ground-truth boxes of the training images
Y_train = encode_batch(y_truth=Y_train, 
                      default=default, 
                      numClasses=numClasses, 
                      input_shape=input_shape, 
                      iou_thres=iou_thres)

print("Shape of parsed training images: {}".format(X_train.shape))
print("Shape of encoded training labels: {}".format(Y_train.shape))

print("Building model...\n")
# Build the SSD model
K.clear_session() # Clear previous session

exist = True

model_path = os.getcwd()
model_path = os.path.join(model_path, 'weights.04-0.38.hdf5')

def iou(Y_true, Y_pred): 
  return iou_metrics(Y_true, Y_pred, default)

if exist: 
  model = load_model(model_path, custom_objects={'loss_function': loss_function, 'iou': iou})
else: 
  # Build the model
  model = build_SSD300(input_shape=input_shape, 
                    numClasses=numClasses, 
                    mode='training', 
                    min_scale=min_scale, 
                    max_scale=max_scale, 
                    aspect_ratios=aspect_ratios, 
                    iou_thres=iou_thres,
                    nms_thres=nms_thres, 
                    score_thres=score_thres, 
                    top_k=top_k,
                    n_predictions=n_predictions)

# Instantiate the Adam optimizer for the model
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0005)

# Create a metrics function


print("Compiling...")
# Compile the model
model.compile(optimizer=adam, loss=loss_function, metrics=['accuracy', iou])


# Path to store learn weights
path = os.getcwd()

checkpoints = ModelCheckpoint(filepath='weights.epoch{epoch:02d}-val_loss{val_loss:.2f}.hdf5', 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=False, 
                             save_weights_only=False, 
                             mode='min', 
                             period=1)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.0,
                               patience=15,
                               verbose=1)

csv_logger = CSVLogger(filename='training_log.csv',
                       separator=',',
                       append=True)

remote = RemoteMonitor()

tensorboard = TensorBoard(log_dir="logs/1557971422.398355", 
                          histogram_freq=0, 
                          batch_size=32, 
                          write_graph=True, 
                          write_grads=True, 
                          write_images=True, 
                          update_freq='batch')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                         factor=0.2,
                                         patience=15, 
                                         min_lr=0.001)

callbacks = [checkpoints, 
            early_stopping, 
            csv_logger,
            tensorboard, 
            reduce_lr, 
            remote]

# Set training parameters
batch_size = 32
initial_epoch = 10
total_epochs = 30

validation_split = 0.2
# When you don't want to train on the entire dataset, use this
# instead of batch_size
steps_per_epoch = 150

history = model.fit(x=X_train, 
                    y=Y_train, 
                    batch_size=batch_size,
                    epochs=total_epochs,
                    verbose=1, 
                    callbacks=callbacks,
                    validation_split=validation_split, 
                    shuffle=True,
                    initial_epoch=initial_epoch)

plt.figure(figsize=(20, 12))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.legend(loc='best')
