from glob import glob
import re
import cv2
from cv2 import imread
import cv2
import pickle

timestamps = []
filenames = []
images = []
sequence_folder = ''
saved_folder = ''

currentIndex = 0

def readFolder(folder, saved):
    global sequence_folder, saved_folder
    sequence_folder = folder
    saved_folder = saved
    with open(f'{sequence_folder}/rgb.txt') as f:
        lines = f.readlines()
        for line in lines[3:]:
            timestamps.append(line.split()[0])
            filenames.append(line.split()[1])
    
    

def getNewFrames(last=len(filenames)):
    readCurrentIndex()
    if last <= currentIndex:
        return [], []
    for filename in filenames[currentIndex:last]:
        images.append(cv2.imread(f'{sequence_folder}/{filename}', cv2.IMREAD_COLOR))
    saveCurrentIndex(last)
    return filenames[currentIndex:last], images



def showVideo():
    timestep = int((30.9/len(filenames))*1000)
    for filename in filenames:
        cv2.imshow('Video Test', cv2.imread(f'{sequence_folder}/{filename}', cv2.IMREAD_COLOR))
        if(cv2.waitKey(timestep) & 0xFF == ord('q')):
            break
    cv2.destroyAllWindows() 

def saveCurrentIndex(index):
    global currentIndex
    currentIndex = index
    with open(f'{saved_folder}/currentIndex.txt', 'w') as indexFile:
        indexFile.write(str(currentIndex))
    

def readCurrentIndex():
    global currentIndex
    with open(f'{saved_folder}/currentIndex.txt', 'r') as indexFile: 
        currentIndex = int(indexFile.read())
    return currentIndex


def saveDescriptors(descriptor_list):
    outfile = open(f'{saved_folder}/descriptor_list.obj', 'wb') 
    pickle.dump(descriptor_list, outfile)


def readDescriptors():
    infile = open(f'{saved_folder}/descriptor_list.obj', 'rb') 
    return pickle.load(infile)


def saveKNN(KNN, scaler, cluster):
    knnFile = open(f'{saved_folder}/knn_model', 'wb') 
    scalerFile = open(f'{saved_folder}/scaler_model', 'wb') 
    clusterFile = open(f'{saved_folder}/cluster_model', 'wb') 
    pickle.dump(KNN, knnFile)  
    pickle.dump(scaler, scalerFile) 
    pickle.dump(cluster, clusterFile)     

def readKNN():
    knnFile = open(f'{saved_folder}/knn_model', 'rb')
    scalerFile = open(f'{saved_folder}/scaler_model', 'rb')
    cluster = open(f'{saved_folder}/cluster_model', 'rb')
    return pickle.load(knnFile), pickle.load(scalerFile), pickle.load(cluster)
