from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import os
import json
import os

class Parser():
    def __init__ (self, data_dir, data_type, resize_shape=None):
        '''
        data_dir: a path to directory containing annotations
        data_type: type of the annotations, train or val
        resize_shape: an array of desired image dimension
        '''
        self.data_dir = data_dir
        self.data_type = data_type
        annFile = '{}/annotations/instances_{}.json'.format(self.data_dir,self.data_type)
        self.coco = COCO(annFile)
        self.resize_shape = resize_shape
        
        

    def parse_json(self):
        '''
        Parser for COCO dataset
        '''
        coco = self.coco

        catIds = [1,2,3,4,5,6,7,8,9,10]
        numClasses = len(catIds)

        self.ground_truth = np.empty(shape = (0, numClasses + 4))
        self.img_ids = []

        if self.resize_shape != None:
            width_ratio = resize_shape[0]
            height_ratio = resize_shape[1]

        ''' No need
        # Get categories 
        cats = coco.loadCats(ids=catIds)
        # person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic
        classes_to_names = [] # A list of the class names with their indices representing the transformed IDs
        classes_to_names.append('background') # Need to add the background class first so that the indexing is right.
        img_ids = []
        for id, cat in cats:
            classes_to_names.append(cat['name'])
        '''
    
        for id in catIds:
            annIds = coco.getAnnIds(catIds = id)
            # labels is 0,1 array, 1 marks the presence of class
            labels = np.zeros(shape = (1, numClasses), dtype='int')
            labels[0][id-1] = 1
            anns = coco.loadAnns(annIds)
            for ann in anns:
                self.img_ids.append(ann['image_id']) 
                img = coco.loadImgs(ann['image_id'])
                width = img[0]['width']
                height = img[0]['height']

                if self.resize_shape == None:
                    width_ratio = width
                    height_ratio = height
                # ann['bbox'] = (xmin, ymin, w, h)
                # bbox = (cx, cy, w, h)
                # If we need to resize image, then divide by width/300 or height/300
                bbox = np.zeros(shape=(1, 4), dtype='float')
                bbox[0][0] = (float) (ann['bbox'][0] + ann['bbox'][2]) / 2 / width  \
                                * (width_ratio / width) # Normalized
                bbox[0][1] = (float) (ann['bbox'][1] + ann['bbox'][3]) / 2 / height \
                                * (height_ratio / height) # Normalized
                bbox[0][2] = (float) (ann['bbox'][2]) / width
                bbox[0][3] = (float) (ann['bbox'][3]) / height
    
                # Create a ground_truth item
                gt = np.append(labels, bbox, axis = 1)
                self.ground_truth = np.append(self.ground_truth, gt, axis = 0) #Shape(#gt, numClasses + 4)

        return self.ground_truth


    def load_img_paths(self, fileName=None):
        '''
        Load all images of ten above categories
        '''

        # Open file for writing (overwrite mode)
        # Change path when running
        path = os.getcwd()
        if(fileName==None): 
            fileName = "imagePaths.txt"
        else: 
            fileName = fileName + ".txt"

        path = path + fileName
    
        f= open(path,"w")

        for imgId in self.img_ids:
            path = '{}/{}/{:012}.jpg\n'.format(self.dataDir, self.dataType, imgId)
            f.write(path)

        f.close()


'''
if __name__ == "__main__":
    data_dir = "/Users/ngophuongnhi/Desktop/csc262proj/cocoapi"
    data_type = "val2017"
    p = Parser(data_dir, data_type)
    gt = p.parse_json()

    print(gt[:10][0] == True)
'''
