import numpy as np 

def generateRandomImages(input_shape=None):
	imgs = np.random.random_sample(input_shape)
	return imgs

def generateLabels(batchSize=1, numElement=1, numClass=1):
	labels = np.random.rand((numClass, 1))

	return labels


# imgs = generateRandomImages((60,300,300,3))
labels = generateLabels(60, 10)
# print(imgs.shape)
print(labels.shape)


