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

KERNEL_DEGREE = 2


#   Load images from the folder path, and returns them as a matrix with the full image as a row.
#   Each image loaded is a row of the matrix.
#   The path directory should have the images grouped in subfolders, where each subfolder is a diferent person.
#   Each person should have IMAGES_PER_PERSON images. 
def imageLoader(path):
    directories = [d for d in listdir(path) if isdir(join(path, d))]

    images = np.zeros([len(directories) * IMAGES_PER_PERSON, IMG_HEIGHT * IMG_WIDTH])
    imgCount = 0
    persons = list()

    for d in directories:
        for imgNo in range(1, IMAGES_PER_PERSON + 1):
            img = ( im.imread(path + d + "/" + str(imgNo) + ".pgm", mode='L') - (MAX_BYTE_VALUE/2) )/ (MAX_BYTE_VALUE/2)
            images[imgCount, :] = np.reshape(img, [1, IMG_HEIGHT * IMG_WIDTH])
            imgCount += 1
            persons.append(d)

    return images, np.asarray(persons)



#   Get Q, R from matrixA.
#   Implemented using the householder method, which is computationaly better than doing gram-schmidt.
#   It also gets Q and R at the same time.
#   This algorithm is O( 2*m*n^2 - (2/3)*n^3 ), while gram-schmidt is O( 2*m*n^2 )
#   The householder algorithm is the most used algorithm for Q, R.
#   MATLAB implement householder
#   Code inspired in:
#       http://www.seas.ucla.edu/~vandenbe/133A/lectures/qr.pdf
#       https://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf
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



#   Get the eigenvalues and eigenvectors of matrixA. 
#   Returns eigenvalues as an array, 
#   and eigenvectors as a matrix where the i column is the eigenvector of the i eigenvalue.
#   The algorithm uses Q, R factorization, and the Q, R algorithm.
#   To get the eigenvector matrix (S), just uses the algorithm:
#       S = Q0
#       Sn+1 = Sn * Qn 
#   This algorithm iterates 100 times the Q, R algorithm, or stops when the result
#   matrix A is diagonal. It's assumed that the matrix is diagonal when the values 
#   outside the diagonal are less than 0.0001
#   Code inspired in:
#       https://www.physicsforums.com/threads/how-do-i-numerically-find-eigenvectors-for-given-eigenvalues.561763/
#       http://www-users.math.umn.edu/~olver/aims_/qr.pdf
def getEigenvaluesAndEigenvectors(matrixA):
    A = np.copy(matrixA).astype(float)
    S = np.eye(matrixA.shape[0])

    for i in range(0, 100): # 100 is the default QR iteration limit
        Q, R = getQandR(A)
        A = np.matmul(R, Q)
        S = np.matmul(S, Q)

        aux = A - np.diag(np.diag(A))
        if np.absolute(aux).max() < 0.0001:
            break

    return np.diag(A), S



#   Get autofaces from matrixA. Autofaces are the eigenvectors from matrixA,
#   divided by the singular values.
#   The singular values are related to the eigenvalues as: singVal = sqrt(eigenvalue)
#   This algorithm stop getting the columns of V when the realted eigenvalues is less
#   than 0.00001
def getAutofaces(matrixA):
    eigenvalues, eigenVectors = getEigenvaluesAndEigenvectors(matrixA)

    n = matrixA.shape[1]
    V = np.zeros([n,n])

    for i in range(0, eigenVectors.shape[1]):
        # Stop when eigenvalue is too small
        if eigenvalues[i] < 0.00001:
            break

        v = np.reshape(eigenVectors[:, i], (-1, 1)) / np.sqrt(eigenvalues[i])  # reshape matrix to avoid shape error
        
        V[i,:] = np.transpose(v)

    return np.transpose(V)
    


#   Generate a centered kernel according to the given code, 
#   and based in the polinomical kernel function described in the PDF of the class:
#       https://s3.us-east-1.amazonaws.com/blackboard.learn.xythos.prod/5a31a0302d72d/206989?response-content-disposition=inline%3B%20filename%2A%3DUTF-8%27%27kernelpres%25281%2529.pdf&response-content-type=application%2Fpdf&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20180913T230835Z&X-Amz-SignedHeaders=host&X-Amz-Expires=21600&X-Amz-Credential=AKIAIL7WQYDOOHAZJGWQ%2F20180913%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=36eb50c2f03214469ce870d272cccdc4c0d273c8544f2f7f8a41087300aacc8c
def generateKernel(matrixA):
    m, n = matrixA.shape
    K = np.zeros([m, m])
    
    for i in range(0, m):
        for j in range(0,m):
            K[i,j] = np.dot( np.transpose(matrixA[i, :]), matrixA[j, :] ) ** KERNEL_DEGREE

    oneDividedM = np.ones([m, m]) / m

    K -= np.dot(oneDividedM, K) - np.dot(K, oneDividedM) + np.dot(oneDividedM, np.dot(K, oneDividedM))

    return K



#   Generate a center kernel for the test image according to the main kernel.
def genereateTestKernel(matrixA, testImg, K):
    m, n = matrixA.shape
    
    testK = np.zeros([1,m])

    for i in range(1, m):
        testK[0, i] = np.dot( testImg, np.transpose(matrixA[i, :]) ) ** KERNEL_DEGREE

    oneDividedM = np.ones([m, m]) / m
    oneDividedMTest = np.ones([1,m]) / m
    testK -= np.dot(oneDividedMTest, K) - np.dot(testK, oneDividedM) + np.dot(oneDividedMTest, np.dot(K, oneDividedM))

    return testK


### Code for facial recognition ### print("Loading images")

print("Loading images")

images, persons = imageLoader("att_faces/")

print("Generating kernel")

K = generateKernel(images)

print("Generating autofaces")

autofaces = getAutofaces(K)

projected = np.transpose(np.dot(autofaces, np.transpose(K)))

# Using support vector machine to predict the test image

print("Loading SVM")

clf = svm.LinearSVC()
clf.fit(projected, persons)

print("Loading test image")

testImg = ( im.imread("att_faces/martin/9.pgm", mode='L') - MAX_BYTE_VALUE/2 ) / (MAX_BYTE_VALUE/2)
testImg = np.reshape(testImg, [1, IMG_HEIGHT * IMG_WIDTH])

testK = genereateTestKernel(images, testImg, K)

projectedTestImg = np.transpose(np.dot(autofaces, np.transpose(testK)))

print("Predicting image")

print("The test image is: " + str(clf.predict(projectedTestImg)[0]))
