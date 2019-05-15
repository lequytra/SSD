from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, LearningRateScheduler, TensorBoard
from keras import backend as K
from keras.callbacks import History, ModelCheckpoint
from keras.models import load_model
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

# Configuration of the model: 
numClasses = 2
nms_thres=0.45 # IoU threshold for non-maximal suppression
score_thres=0.01 # threshold for classification scores
top_k=200 # the maximum number of predictions kept per image

min_scale=0.2 # the smallest scale of the feature map
max_scale=0.9 # the largest scale of the feature map
aspect_ratios=[0.5, 1, 2] # aspect ratios of the default boxes to be generated
n_predictions=6 # the number of prediction blocks
prediction_size=[38, 19, 10, 5, 3, 1] # sizes of feature maps at each level

# Load data for prediction
data_dir = "/Users/ngophuongnhi/Desktop/csc262proj/cocoapi"
predicting_data = "val2017"
# Initialize a parser object
parser = Parser(data_dir, predicting_data, numClasses)

imgs, labels = parser.load_data() # labels (batch_size, n_boxes, 1 + numClasses + 4)

# Load trained model
model_path = '/Users/ngophuongnhi/Downloads/weights.03-5.21.hdf5' #No model path

K.clear_session() # Clear previous session

defaults = generate_default_boxes(n_layers=n_predictions, 
                                min_scale=min_scale, 
                                max_scale=max_scale, 
                                map_size=prediction_size, 
                                aspect_ratios=aspect_ratios)

# Create a metrics function
def iou(Y_true, Y_pred): 
	return iou_metrics(Y_true, Y_pred, defaults)

# Load previous model
model = load_model(model_path, custom_objects={'loss_function': loss_function,
											'iou': iou})

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)



# Compile the model
model.compile(optimizer=adam, loss=loss_function, metrics=['accuracy', iou])


# Predict object detection
pred = model.predict(imgs[0])

# Decode prediction result
pred_decoded = Decoder(predictions=y_pred, 
                        defaults=defaults, 
                        numClasses=numClasses, 
                        nms_thres=np.float32(0), 
                        score_thres=np.float32(0), 
                        top_k=top_k)

# decoder returns (top_k, label score x1 y1 x2 y2)
# Get the class_id with highest scores (n_boxes, label, 4)
pred_label = pred_decoded.prediction_out()

''' Decode wrong not working
colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic']

plt.figure(figsize=(20,12))
i = 0
plt.imshow(imgs[i])

current_axis = plt.gca()

for box in labels[i]:
    xmin = box[-4]
    ymin = box[-3]
    xmax = box[-2]
    ymax = box[-1]
    label_id = [i for i,e in enumerate(box(:-4)) if e != 0]# Find the correct class where it is not zero
    label = '{}'.format(classes[int(label_id[0])])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

for box in pred_label[i]:
    xmin = box[-4]
    ymin = box[-3]
    xmax = box[-2]
    ymax = box[-1]
    label_id = [i for i,e in enumerate(box(:-4)) if e != 0]# Find the correct class where it is not zero
    color = colors[int(label_id[0])]
    label = '{}: {:.2f}'.format(classes[int(label_id[0])], y_pred_label[i][label_id[0]])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
    '''