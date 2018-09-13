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


'''
   Load images from the folder path, and returns them as a matrix with the full image as a row.
   Each image loaded is a row of the matrix.
   The path directory should have the images grouped in subfolders, where each subfolder is a diferent person.
   Each person should have IMAGES_PER_PERSON images. 
'''
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



'''
   Get Q, R from matrixA.
   Implemented using the householder method, which is computationaly better than doing gram-schmidt.
   It also gets Q and R at the same time.
   This algorithm is O( 2*m*n^2 - (2/3)*n^3 ), while gram-schmidt is O( 2*m*n^2 )
   The householder algorithm is the most used algorithm for Q, R.
   MATLAB implement householder
'''
def getQandR(matrixA):
    m, n = matrixA.shape
    Q = np.eye(m)
    R = np.copy(matrixA).astype(float) #change type to float to avoid default int errors

    for j in range(0, n):
        normx = np.linalg.norm(R[j:, j])
        s = -np.sign(R[j,j])
        u1 = R[j,j] - s * normx
        w = np.reshape(R[j:,j], (-1, 1)) / u1  #reshape matrix to avoid size error
        w[0] = 1
        tau = -s * u1 / normx
        
        wTxR = np.reshape( np.dot(np.transpose(w), R[j:,:]), (1, -1) )  #reshape matrix to avoid size error

        R[j:,:] -= np.matmul((tau * w), wTxR)
        Q[:,j:] -= np.dot( np.dot(Q[:, j:], w) , np.transpose(tau * w) )

    return Q, R 



'''
   Get the eigenvalues and eigenvectors of matrixA. 
   Returns eigenvalues as an array, 
   and eigenvectors as a matrix where the i column is the eigenvector of the i eigenvalue.
   The algorithm uses Q, R factorization, and the Q, R algorithm.
   To get the eigenvector matrix (S), just uses the algorithm:
       S = Q0
       Sn+1 = Sn * Qn 
'''
def getEigenvaluesAndEigenvectors(matrixA):
    A = np.copy(matrixA).astype(float)
    S = np.eye(matrixA.shape[0])

    for i in range(0,1500): #choose correct range, 1500 is just for testing
        Q, R = getQandR(A)
        A = np.matmul(R, Q)
        S = np.matmul(S, Q)

    return np.diag(A), S






# Code of facial recognition working, using SVD
'''
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

'''



# Debuging Eigenvalues and Eigenvectors

A = np.array([[2, 1, 2], [2, 2, 1], [1, 2, 2]])
#Q, R = getQandR(A)
eigenvalues, eigenvectors = getEigenvaluesAndEigenvectors(A)


print(A)
print(eigenvalues)
print(eigenvectors)