import cv2
import sys
from os import listdir
from os.path import join, isdir
from scipy import ndimage as im
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import math


IMAGES_PER_PERSON = 8
TEST_IMAGES_PER_PERSON = 2
MAX_BYTE_VALUE = 255.0
IMG_HEIGHT = 112
IMG_WIDTH = 92



### Code for facial recognition ###

def imageLoader(path):
    directories = [d for d in listdir(path) if isdir(join(path, d))]

    images = np.zeros([len(directories) * IMAGES_PER_PERSON, IMG_HEIGHT * IMG_WIDTH])
    imgCount = 0
    persons = list()

    for d in directories:
        for imgNo in range(1, IMAGES_PER_PERSON + 1):
            img = im.imread(path + d + "/" + str(imgNo) + ".pgm", mode='L')/MAX_BYTE_VALUE
            images[imgCount, :] = np.reshape(img, [1, IMG_HEIGHT * IMG_WIDTH])
            imgCount += 1
            persons.append(d)

    return images, np.asarray(persons)

def testImageLoader(path):
    directories = [d for d in listdir(path) if isdir(join(path, d))]
    images = np.zeros([len(directories) * TEST_IMAGES_PER_PERSON, IMG_HEIGHT * IMG_WIDTH])
    imgCount = 0
    persons = list()

    for d in directories:
        for imgNo in range(IMAGES_PER_PERSON + 1, TEST_IMAGES_PER_PERSON +IMAGES_PER_PERSON+ 1):
            img = im.imread(path + d + "/" + str(imgNo) + ".pgm", mode='L')/MAX_BYTE_VALUE
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


#   According to SVD, a matrix A = U * S * V'
#   This function returns V'
#   The columns of V are the eigenvectors of A'A, this matrix 
#   is to big to get it's eigenvectors, so they are calculated using U
#   The columns of U are the eignevectors of AA'
#   v = A' * u / singVal, where v are the columns of V, u are the columns of U,
#   singVal are the singular values from the diagonal matrix S
#   The singular values are related to the eigenvalues as: singVal = sqrt(eigenvalue)
#   This algorithm stop getting the columns of V when the realted eigenvalues is less
#   than 0.00001
def getVofSVD(matrixA):
    At = np.transpose(matrixA)
    AAt = np.matmul(matrixA, At)

    eigenvalues, U = getEigenvaluesAndEigenvectors(AAt) # U are the eigenvectors

    n = matrixA.shape[1]
    V = np.zeros([U.shape[1],n])

    for i in range(0, U.shape[1]):

        # Stop when eigenvalue is too small
        if eigenvalues[i] < 0.00001:
            break

        u = np.reshape(U[:, i], (-1, 1))  # reshape matrix to avoid shape error
        v = np.matmul(At, u) / np.sqrt(eigenvalues[i])
        
        V[i,:] = np.transpose(v)

    return np.transpose(V)
    



print("Loading images")

images, persons = imageLoader("att_faces/")

meanImage = np.mean(images, 0)
images -= meanImage
print("Generating autofaces")

#U, S, V = np.linalg.svd(images, full_matrices=False)
V = getVofSVD(images)

print(V.shape)

projected = np.dot(images, V)

# Using support vector machine to predict the test image

print("Loading SVM")

clf = svm.LinearSVC()
clf.fit(projected, persons)




cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
framewidth = video_capture.get(3)   # float
frameheight = video_capture.get(4)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(92, 112)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        croph  = int(max(0,y-h/2))
        crophm = int(min(frameheight,y+h+h/2))

        cropw  = int(max(0,x-w/2)) 
        cropwm = int(min(framewidth,x+w+w/2))


        crop_img = frame[y:y+h, x:x+w]
        #crop_img = frame[croph:crophm, cropw:cropwm]
        #cv2.imshow('img',crop_img)
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        crop_img2 = cv2.resize(crop_img,(int(IMG_WIDTH),int(IMG_HEIGHT)))

        crop_img = np.reshape(crop_img2, [1, IMG_HEIGHT * IMG_WIDTH]) 
        testImg = crop_img - meanImage
        projectedTestImg = np.dot(testImg, V)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
        name = str(clf.predict(projectedTestImg)[0])
        print("The test image is: " + name+"h: "+str(x)+" w: "+str(y))
        cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        

    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()