import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.cluster import KMeans

def runBoW(keypoints, descriptors):
    kmeans = KMeans(n_clusters = 1)
    kmeans.fit(descriptors)
    histogram = build_histogram(descriptors, kmeans)
    print(histogram)
    pass