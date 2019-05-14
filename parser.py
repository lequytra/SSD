from pycocotools.coco import COCO
import numpy as np
import keras.backend as K
import tensorflow as tf
from tensorflow.image import draw_bounding_boxes
import cv2
import matplotlib.pyplot as plt
import os
import json

class Parser():
    def __init__ (self, data_dir, data_type, resize_shape=(300,300,3)):
        '''
        data_dir: a path to directory containing annotations
        data_type: type of the annotations, train or val
        resize_shape: an array of desired image dimension
        '''
        self.data_dir = data_dir
        self.data_type = data_type
        annFile = '{}/annotations/instances_{}.json'.format(self.data_dir,self.data_type)
        self.coco = COCO(annFile)
        self.fileName = None
        self.resize_shape = resize_shape
        
        

    def parse_json(self):
        '''
        Parser for COCO dataset

        '''
        coco = self.coco

        numClasses = 10

        cat_ids = [i + 1 for i in range(numClasses)]
        numClasses = len(cat_ids)

        self.ground_truth = []
        self.img_ids = set()

        if self.resize_shape != None:
            width_ratio = self.resize_shape[0]
            height_ratio = self.resize_shape[1]
    
        # Get all images from chosen categories, prevent duplication
        for cat_id in cat_ids:
            images = coco.getImgIds(catIds=cat_id)
            for img_id in images:
                self.img_ids.add(img_id)

        for img_id in self.img_ids: 
            batch = np.empty(shape=(0, numClasses + 4), dtype='float')
            img = coco.loadImgs(img_id)

            # Get image dimension
            width = img[0]['width']
            height = img[0]['height']

            if self.resize_shape == None:
                width_ratio = width
                height_ratio = height

            for cat_id in cat_ids:
                # labels is 0,1 array, 1 marks the presence of class
                labels = np.zeros(shape = (1, numClasses), dtype='int')
                labels[0][cat_id-1] = 1                

                ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_id)

                # Loop through annotations for each image
                for ann_id in ann_ids:
                    ann = coco.loadAnns(ann_id)
                    # print(ann[0]['bbox'])
                     # ann_id['bbox'] = (xmin, ymin, w, h)
                    bbox = np.zeros(shape=(1,4), dtype='float')
                    bbox[0][0] = (float) (ann[0]['bbox'][0] / width)  * (width_ratio / width) # Normalized
                    bbox[0][1] = (float) (ann[0]['bbox'][1] / height) * (height_ratio / height) # Normalized
                    bbox[0][2] = (float) (ann[0]['bbox'][2] / width) * (width_ratio / width)
                    bbox[0][3] = (float) (ann[0]['bbox'][3] / height) * (height_ratio / height)

                    # Create a ground_truth item
                    gt = np.append(labels, bbox, axis=1)

                    batch = np.append(batch, gt, axis=0)

            self.ground_truth.append(batch)

        return self.ground_truth


    def load_img_paths(self, fileName=None):
        '''
        Load all images of ten above categories
        '''

        # Open file for writing (overwrite mode)
        # Change path when running
        path = os.getcwd()

        if(fileName==None): 
            self.fileName = "imagePaths.txt"
        else: 
            self.fileName = fileName + ".txt"

        path = os.path.join(path, self.fileName)

        f= open(path,"w+")

        for imgId in self.img_ids:
            im_path = '{}/{}/{:012}.jpg\n'.format(self.data_dir, self.data_type, imgId)
            f.write(im_path)

        f.close()

    def load_training_imgs(self):
        path = os.getcwd()
        path = os.path.join(path, self.fileName)

        im_collection = []

        with open(path) as f: 

            for im_path in f: 

                im_path = im_path.strip()
                image = cv2.imread(im_path)

                image = cv2.resize(image, dsize=(self.resize_shape[0], 
                                                self.resize_shape[1]))

                im_collection.append(image)
            
        return np.array(im_collection)

    def load_data(self): 
        """
            Output: 
                Return a tuple of image, label
        """
        label = self.parse_json()
        self.load_img_paths()
        im_collection = self.load_training_imgs()

        return (im_collection, label)


def main(): 
    data_dir = "/Users/tranle/mscoco"
    training_data = "val2017"
    # Initialize a parser object
    parser = Parser(data_dir, training_data)

    # Load images and annotations for the image
    # For now, we load only 10 first classes and images are resize to (300,300,3) 
    # for training purposes

    X, Y = parser.load_data()

    # X = np.array(X)
    Y = np.array(Y)

    print("Shape of parsed images: {}".format(X.shape))
    print("Shape of parsed labels: {}".format(Y.shape))
    print("Shape of one label: {}".format(Y[0].shape))




# if __name__ == "__main__":
#     # main()
