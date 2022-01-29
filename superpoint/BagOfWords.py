import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.cluster import KMeans
from sklearn import svm

def runBoW(keypoints, descriptors):
    read_and_clusterize(descriptors, 1)
    # kmeans = KMeans(n_clusters = 1)
    # kmeans.fit(descriptors)
    # histogram = build_histogram(descriptors, kmeans)
    # print(histogram)




#this function will get SIFT descriptors from training images and 
#train a k-means classifier    
def read_and_clusterize(descriptors, num_clusters):

    sift_keypoints = []
    sift_keypoints.append(descriptors)

    sift_keypoints = np.asarray(sift_keypoints)
    sift_keypoints = np.concatenate(sift_keypoints, axis=0)
    #with the descriptors detected, lets clusterize them
    print("Training kmeans")    
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(sift_keypoints)
    #return the learned model
    return kmeans

# #with the k-means model found, this code generates the feature vectors 
# #by building an histogram of classified keypoints in the kmeans classifier 
# def calculate_centroids_histogram(file_images, model):

#     feature_vectors=[]
#     class_vectors=[]

#     with open(file_images) as f:
#         images_names = f.readlines()
#         images_names = [a.strip() for a in images_names]

#         for line in images_names:
#         print(line)
#         #read image
#         image = cv2.imread(line,1)
#         #Convert them to grayscale
#         image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#         #SIFT extraction
#         sift = cv2.xfeatures2d.SIFT_create()
#         kp, descriptors = sift.detectAndCompute(image,None)
#         #classification of all descriptors in the model
#         predict_kmeans=model.predict(descriptors)
#         #calculates the histogram
#         hist, bin_edges=np.histogram(predict_kmeans)
#         #histogram is the feature vector
#         feature_vectors.append(hist)
#         #define the class of the image (elephant or electric guitar)
#         class_sample=define_class(line)
#         class_vectors.append(class_sample)

#     feature_vectors=np.asarray(feature_vectors)
#     class_vectors=np.asarray(class_vectors)
#     #return vectors and classes we want to classify
#     return class_vectors, feature_vectors


# def define_class(img_patchname):

#     #print(img_patchname)
#     print(img_patchname.split('/')[4])

#     if img_patchname.split('/')[4]=="electric_guitar":
#         class_image=0

#     if img_patchname.split('/')[4]=="elephant":
#     class_image=1

#     return class_image

# def main(train_images_list, test_images_list, num_clusters):
#     #step 1: read and detect SURF keypoints over the input image (train images) and clusterize them via k-means 
#     print("Step 1: Calculating Kmeans classifier")
#     model= bovw.read_and_clusterize(train_images_list, num_clusters)

#     print("Step 2: Extracting histograms of training and testing images")
#     print("Training")
#     [train_class,train_featvec]=bovw.calculate_centroids_histogram(train_images_list,model)
#     print("Testing")
#     [test_class,test_featvec]=bovw.calculate_centroids_histogram(test_images_list,model)

#     #vamos usar os vetores de treino para treinar o classificador
#     print("Step 3: Training the SVM classifier")
#     clf = svm.SVC()
#     clf.fit(train_featvec, train_class)

#     print("Step 4: Testing the SVM classifier")  
#     predict=clf.predict(test_featvec)

#     score=accuracy_score(np.asarray(test_class), predict)

#     file_object  = open("results.txt", "a")
#     file_object.write("%f\n" % score)
#     file_object.close()

#     print("Accuracy:" +str(score))

# if __name__ == "__main__":
#     main("train.txt", "test.txt", 1000)
#     main("train.txt", "test.txt", 2000)
#     main("train.txt", "test.txt", 3000)
#     main("train.txt", "test.txt", 4000)
#     main("train.txt", "test.txt", 5000)