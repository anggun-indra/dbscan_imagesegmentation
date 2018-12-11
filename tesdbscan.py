import cv2
import scipy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


img = cv2.imread('toll1.jpg') #('buoy1.png')
labimg = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
n = 0
while(n<4):
    labimg = cv2.pyrDown(labimg)
    n = n+1

indices = np.dstack(np.indices(img.shape[:2]))
xycolors = np.concatenate((img, indices), axis=-1) 
feature_image=np.reshape(xycolors, [-1,5])
rows, cols, chs = labimg.shape

db = DBSCAN(eps=5, min_samples=50, metric = 'euclidean',algorithm ='auto')
db.fit(feature_image)
labels = db.labels_
plt.figure(2)
plt.subplot(2, 1, 1)
plt.imshow(img)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(np.reshape(labels, [1, 2]))
plt.axis('off')
plt.show()
