from glob import glob
import re
import cv2
from cv2 import imread
import cv2
import pickle
import numpy as np

timestamps = []
filenames = []
images = []
sequence_folder = '/root/ORB_FR1'
saved_folder = 'saved'

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
    
    

def getNewFrames(last=len(filenames),shouldsave=True):
    readCurrentIndex()
    if last <= currentIndex:
        return [], []
    for filename in filenames[currentIndex:last]:
        images.append(cv2.imread(f'{sequence_folder}/{filename}', cv2.IMREAD_COLOR))
    if shouldsave:
        saveCurrentIndex(last)
    return filenames[currentIndex:last], images

def getAllFrames(last=len(filenames)):
    imgs = []
    for filename in filenames[:last]:
        imgs.append(cv2.imread(f'{sequence_folder}/{filename}', cv2.IMREAD_COLOR))
    return imgs

def getLoopClosureFrames(indices):
    imgs = []
    for i in range(len(indices)):
        imgs.append(cv2.imread(f'{sequence_folder}/{filenames[indices[i]]}', cv2.IMREAD_COLOR))
    return imgs

def getFrame(index):
    return cv2.imread(f'{sequence_folder}/{filenames[index]}', cv2.IMREAD_COLOR)

def showFrame(frame):
    cv2.imshow('Frame', frame)

def showLoopClosures(frames):
    i = 0
    while i < len(frames):
        showFrame(frames[i])
        if (cv2.waitKey(1000) & 0xFF == ord('n')):
            i+=1

def showLoopClosurePairs(lcc):
    i = 0
    while i < len(lcc):
        pair = lcc[i]
        img1 = getFrame(pair[0])
        img2 = getFrame(pair[1])
        combined = np.hstack((img1, img2))
        name = f'Loop Closure between frame {pair[0]} and {pair[1]} '
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 640*2, 480)
        cv2.imshow(name, combined)
        out = cv2.waitKey(0)
        if (out & 0xFF) == ord('n'):
            i+=1
            cv2.destroyWindow(name)
        if (out & 0xFF) == ord('q'):
            break

def showVideo():
    timestep = int((30.9/len(filenames))*1000)
    for filename in filenames:
        cv2.imshow('Video', cv2.imread(f'{sequence_folder}/{filename}', cv2.IMREAD_COLOR))
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


def readDescriptors(max=100000):
    infile = open(f'{saved_folder}/descriptor_list.obj', 'rb')
    desc_list = pickle.load(infile)
    if len(desc_list) < max:
        return desc_list
    else:
        return desc_list[:max]


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

def saveLoopClosures(loop_closures, loop_closure_frames):
    with open(f'{saved_folder}/loop_closures.txt', 'w') as lcFile:
        lcFile.write(str(loop_closures))
    with open(f'{saved_folder}/loop_closure_frames.txt', 'w') as lciFile:
        lciFile.write(str(loop_closure_frames))

def readLoopClosures():
    with open(f'{saved_folder}/loop_closures.txt', 'r') as lcFile:
        arr = lcFile.read()[1:-1].replace(' ', '').split(',')
        return [True if x == 'True' else False for x in arr]

def saveLCC(lcc):
    lccFile = open(f'{saved_folder}/lcc', 'wb') 
    pickle.dump(lcc, lccFile)     

def readLCC():
    lcc = open(f'{saved_folder}/lcc', 'rb')
    return pickle.load(lcc)
