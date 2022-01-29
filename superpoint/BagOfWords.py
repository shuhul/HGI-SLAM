import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.cluster import KMeans
import cv2
import matplotlib as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


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


def predict_similar_images(neighbors, image_bow, scaler):
    image_bow = scaler.transform(image_bow.reshape(1,-1))
	# returns the similar images indices along with their distances from the bag of words of the similar images.
    return neighbors.kneighbors(image_bow.reshape(1,-1))


def runBoW(descriptor_list):

    # generate the clusters of the patches
    clusters = create_patch_clusters(np.vstack(descriptor_list), num_clusters=100)

    # convert images into bag of visual words
    image_bows = convert_images_to_bows(descriptor_list, clusters)

    # normalize the bag of words for k-nearest neighbor
    scaler, normalized_bows = scale_bows(image_bows)

    # train the k-NN
    neighbors = train_knn(normalized_bows, 5, 1)

    # determine similar images 
    print(predict_similar_images(neighbors, image_bows[0],scaler))