from pycocotools.coco import COCO
import numpy as np

dataDir='/home/ngonhi/csc262/project/coco'
dataType = 'val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco = COCO(annFile)

# Store ids for all categories in a list
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]

# Load all image ids from interesting categories
catIds = coco.getCatIds(catNms=['person']);
imgIds = coco.getImgIds(catIds=catIds );
count = 0;

# Open file for writing (overwrite mode)
f= open("/home/ngonhi/csc262/project/SSD/imagePaths.txt","w")
for imgId in imgIds:
	path = '{}/{}/{:012}.jpg\n'.format(dataDir, dataType, imgId)
	f.write(path)

f.close()



