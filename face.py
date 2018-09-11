from os import listdir
from os.path import join, isdir
from scipy import ndimage as im
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

IMAGES_PER_PERSON = 8
MAX_BYTE_VALUE = 255.0
IMG_HEIGHT = 112
IMG_WIDTH = 92

def imageLoader(path):
    directories = [d for d in listdir(path) if isdir(join(path, d))]

    images = np.zeros([len(directories) * IMAGES_PER_PERSON, IMG_HEIGHT * IMG_WIDTH])
    imgCount = 0
    persons = list()

    for d in directories:
        for imgNo in range(1, IMAGES_PER_PERSON + 1):
            img = im.imread(path + d + "/" + str(imgNo) + ".pgm")/MAX_BYTE_VALUE
            images[imgCount, :] = np.reshape(img, [1, IMG_HEIGHT * IMG_WIDTH])
            imgCount += 1
            persons.append(d)

    return images, np.asarray(persons)


images, persons = imageLoader("facesvddem/")


meanImage = np.mean(images, 0)
images -= meanImage

U, S, V = np.linalg.svd(images, full_matrices=False)


projected = np.dot(images, np.transpose(V))


clf = svm.LinearSVC()
clf.fit(projected, persons)

testImg = im.imread("facesvddem/s3/10.pgm")/MAX_BYTE_VALUE
testImg = np.reshape(testImg, [1, IMG_HEIGHT * IMG_WIDTH])

testImg -= meanImage

projectedTestImg = np.dot(testImg, np.transpose(V))

print(clf.predict(projectedTestImg))

