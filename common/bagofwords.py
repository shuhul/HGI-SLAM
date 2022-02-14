import numpy as np
from sklearn import neighbors
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.cluster import KMeans
import cv2
import matplotlib as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import common.handler as handler


def create_patch_clusters(descriptor_list, num_clusters):  
    clusters = KMeans(n_clusters = num_clusters, init="k-means++")
    clusters.fit(np.vstack(descriptor_list))
    return clusters


def create_bag_of_visual_words(descriptors, clusters):
    image_bow = np.zeros(clusters.get_params()['n_clusters'])
    for i in range(0, len(descriptors)):
        cluster = clusters.predict(descriptors[i].reshape(1,-1))
        image_bow[cluster] = image_bow[cluster] + 1
	# returns the bag of words representation of a single image
    return image_bow


def convert_images_to_bows(descriptor_list, clusters):
    image_bows = []
    for i in range(0, len(descriptor_list)):
        image_bows.append(create_bag_of_visual_words(descriptor_list[i], clusters))
    image_bows = np.vstack(image_bows)
	# returns the bag of words of each image 
    return image_bows


def scale_bows(image_bows):
    sc = StandardScaler()
    image_bows_normalized = sc.fit_transform(image_bows)
	# returns a StandardScaler which can be used to normalize query image (represented by bag of words)
	# and also returns the normalized bag of words of all images
    return sc, image_bows_normalized


def train_knn(image_bows, n_neighbors=5, radius=1):
    neighbors = NearestNeighbors(n_neighbors = n_neighbors, radius = radius, n_jobs = 2)
    neighbors.fit(image_bows)
	# returns the k-NN model
    return neighbors


def getSimilarBoW(descriptor):
    neighbors, scaler, cluster = handler.readKNN()

    image_bow = convert_images_to_bows(descriptor_list=[descriptor], clusters=cluster)

    image_bow = scaler.transform(image_bow.reshape(1,-1))
	# returns the similar images indices along with their distances from the bag of words of the similar images.
    output = neighbors.kneighbors(image_bow.reshape(1,-1))[::-1]
    return output[0][0], output[1][0]


def trainBoW(descriptor_list, n_clusters, n_neighbors):

    print(f'Generating {n_clusters} clusters')
    clusters = create_patch_clusters(np.vstack(descriptor_list), num_clusters=n_clusters)

    
    image_bows = convert_images_to_bows(descriptor_list, clusters)

    scaler, normalized_bows = scale_bows(image_bows)

    
    print(f'Training KNN with {n_neighbors} neighbors')

    neighbors = train_knn(normalized_bows, n_neighbors)

    handler.saveKNN(neighbors, scaler, clusters)


def isLoopClosure(distance, min_distance):
    return distance < min_distance


def detectLoopClosures(descriptor_list, min_distance):
    lcc = []
    for i in range(len(descriptor_list)):
        indices, distances = getSimilarBoW(descriptor_list[i])
        for index, distance in zip(indices, distances):
            if index != i and isLoopClosure(distance, min_distance):
                lcc.append([i, index])
                # loop_closures[i] = True
    loop_closures = [False] * len(descriptor_list)
    lcc = removeLCCDups(lcc)
    handler.saveLCC(lcc)
    for lc in lcc:
        loop_closures[lc[1]] = True
    handler.saveLoopClosures(loop_closures)
                
def getLoopClosures():
    loop_closures = handler.readLoopClosures()
    loop_closure_indices = []
    for i in range(len(loop_closures)):
        if loop_closures[i]:
            loop_closure_indices.append(i)
    return loop_closure_indices

def removeLCCDups(lcc):
    res = []
    [res.append(x) for x in lcc if x[::-1] not in res]
    return res

def getLCC():
    return handler.readLCC()