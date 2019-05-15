import cv2
import os
import numpy as np

def load_img(fileName):
        path = os.getcwd()
        path = os.path.join(path, fileName)

        im_collection = []

        with open(path) as f: 

            for im_path in f: 

                im_path = im_path.strip()
                image = cv2.imread(im_path)

                # 

                im_collection.append(image)
            
        return np.array(im_collection)

def resize_im(im_collection, input_shape=(300, 300, 3)): 
    result = np.empty(shape=(0, input_shape[0], input_shape[1], input_shape[2]))
    for im in im_collection: 
        image = cv2.resize(im, dsize=(input_shape[0],input_shape[1]))
        image = np.expand_dims(image, axis=0)
        result = np.append(result, image, axis=0)

    return result