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

        cat_ids = [1,2,3,4,5,6,7,8,9,10]
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
                    #print(ann[0]['bbox'])
                    # ann_id['bbox'] = (xmin, ymin, w, h)
                    bbox = np.zeros(shape=(1,4), dtype='float')
                    bbox[0][0] = (float) (ann[0]['bbox'][0] + ann[0]['bbox'][2]) / 2 / width  * (width_ratio / width) # Normalized
                    bbox[0][1] = (float) (ann[0]['bbox'][1] + ann[0]['bbox'][3]) / 2 / height * (height_ratio / height) # Normalized
                    bbox[0][2] = (float) (ann[0]['bbox'][2]) / width * (width_ratio / width)
                    bbox[0][3] = (float) (ann[0]['bbox'][3]) / height * (height_ratio / height)

                    # Create a ground_truth item
                    gt = np.append(labels, bbox, axis=1)
                    batch = np.append(batch, gt, axis=0)
            self.ground_truth.append(batch)
        return self.ground_truth


    def load_img_paths(self, fileName=None, type="local"):
        '''
        Load all images of ten above categories
        '''

        # Open file for writing (overwrite mode)
        # Change path when running
        if type=="local":
            path = os.getcwd()

            if(fileName==None): 
                self.fileName = "imagePaths.txt"
            else: 
                self.fileName = fileName + ".txt"

            path = path + '/' + self.fileName
            f= open(path,"w+")

            for imgId in self.img_ids:
                im_path = '{}/{}/{:012}.jpg\n'.format(self.data_dir, self.data_type, imgId)
                f.write(im_path)

            f.close()
        elif type=="online":
            path = os.getcwd()

            if(fileName==None): 
                self.fileName = "imagePaths.txt"
            else: 
                self.fileName = fileName + ".txt"

            path = path + '/' + self.fileName
            f= open(path,"w+")

            for imgId in self.img_ids:
                img = coco.loadImgs(imgId)
                im_path = img[0]['coco_url']
                f.write(im_path)

            f.close()

    def load_training_imgs(self):
        path = os.getcwd()
        path = path + '/' + self.fileName

        im_collection = []

        with open(path) as f: 

            for im_path in f: 

                im_path = im_path.strip()
                image = cv2.imread(im_path)

                image = cv2.resize(image, dsize=(self.resize_shape[0], 
                                                self.resize_shape[1]))

                im_collection.append(image)
            
        return im_collection

    def load_data(self, type="local"): 
        """
            Output: 
                Return a tuple of image, label
        """
        label = self.parse_json()
        self.load_img_paths(type)
        im_collection = self.load_training_imgs()

        return (im_collection, label)


def main(): 
    data_dir = "/Users/ngophuongnhi/Desktop/csc262proj/cocoapi"
    #data_dir = "/Users/tranle/mscoco"
    data_type = "val2017"
    p = Parser(data_dir, data_type)
    gt = p.parse_json()

    p.load_img_paths()

    im_collection = p.load_training_imgs()
    bbox_im = [im_collection[:10]]
    bbox_coord = gt[:10][-4:]
    print(type(im_collection))
    print(len(im_collection))
    images = tf.constant(im_collection)
    bbox_coord[:,2] += bbox_coord[:, 0]
    bbox_coord[:, 3] += bbox_coord[:, 1]

    coords = tf.constant(bbox_coord)

    plt.imshow(images[0])




if __name__ == "__main__":
    main()
