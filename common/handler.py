from glob import glob
import re
import cv2
from cv2 import imread
import cv2
import pickle
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import shutil
import math
import os


timestamps = []
filenames = []
sequence_folder = '/root/ORB_FR1'
saved_folder = 'saved'

currentIndex = 0

origin = (1.4, 1.7)
theta = np.deg2rad(-40)
scale = 0.6
    

def readFolder(folder):
    global sequence_folder, saved_folder
    sequence_folder = folder
    saved_folder = sequence_folder[6:]
    doesSavedExist = os.path.exists(saved_folder)
    if not doesSavedExist:
        os.makedirs(saved_folder)
    with open(f'{sequence_folder}/rgb.txt') as f:
        lines = f.readlines()
        for line in lines[3:]:
            timestamps.append(line.split()[0])
            filenames.append(line.split()[1])
    
    

def getNewFrames(last=len(filenames),skip=4):
    global currentIndex
    images = []
    if last == -1:
        last=len(filenames)
    readCurrentIndex()
    if last <= currentIndex:
        return [], [], last
    files = filenames[currentIndex:last:skip]
    for filename in files:
        images.append(cv2.imread(f'{sequence_folder}/{filename}', cv2.IMREAD_COLOR))

    return files, images, last

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

def getFrameNumber(timestamp):
    for i in range(len(timestamps)):
        print(timestamps[i][:-8])
        print(timestamp[:-6])
        if timestamps[i][:-8] == timestamp[:-6]:
            return i
    return 0

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

def showVideo(skip=4):
    timestep = int((30.9/len(filenames))*1000)
    for filename in filenames[::skip]:
        name = 'Video'
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 640, 480)
        cv2.imshow(name, cv2.imread(f'{sequence_folder}/{filename}', cv2.IMREAD_COLOR))
        if(cv2.waitKey(timestep) & 0xFF == ord('q')):
            break
    cv2.destroyAllWindows()

def showTrajectory(showGT = True, create = False):
    global origin, theta, scale

    if create:
        print("\nComputing trajectory\n")
        black = col(33,41,48, 0.7)
        lightblue = col(0, 53, 178, 0.8)
        red = col(220, 32, 52, 0.9)
        green = col(11, 68, 31, 0.95)


        txs, tzs = readGT()
        if showGT:
            plt.plot(txs, tzs, color=black, linewidth=1.0)

        num = copyKFT()

        offset = (-0.1,-0.3)
        origin = (txs[0]+offset[0], tzs[0]+offset[1])
        theta = np.deg2rad(130)
        scale = 0.95

        axs, azs = readKFT('after')
        
        lcInd = 0
        for i in range(1,num):
            bxs, bzs = readKFT(f'before{i}')
            plt.plot(bxs[lcInd:], bzs[lcInd:], color=wcol(red, lightblue, ((i-1)/(num-1))), linewidth=2.0)
            lcInd = len(bxs)
            if i < num-1:
                bxs1, bzs1 = readKFT(f'before{i+1}')
                lcxs = [bxs[-1],bxs1[lcInd]]
                lczs = [bzs[-1],bzs1[lcInd]]
                plt.plot(lcxs, lczs, color=green, linewidth=3.0)
            else:
                lcxs = [bxs[-1],axs[lcInd]]
                lczs = [bzs[-1],azs[lcInd]]
                plt.plot(lcxs, lczs, color=green, linewidth=3.0)
        plt.plot(axs[lcInd:], azs[lcInd:], color=red, linewidth=2.0)
            
        
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.savefig(f'{saved_folder}/kft.png')
    else:
        print("\nSkipping already computed trajectory\n")
    print(f"Displaying HGI-SLAM Trajectory for {saved_folder}\n")
    name = f'HGI-SLAM Trajectory'
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 640, 560)
    cv2.imshow(name, cv2.imread(f'{saved_folder}/kft.png', cv2.IMREAD_COLOR))
    while True:
        if(cv2.waitKey(0) & 0xFF == ord('q')):
            break

def col(r, g, b, a):
    return list(np.array([r, g, b, a*255])/255)

def wcol(col1, col2, w):
    return list(((w)*np.array(col1))+((1-w)*np.array(col2)))

def saveCurrentIndex(index):
    global currentIndex
    currentIndex = index
    with open(f'{saved_folder}/currentIndex.txt', 'w') as indexFile:
        indexFile.write(str(currentIndex))
    

def readCurrentIndex():
    global currentIndex
    if not os.path.exists(f'{saved_folder}/currentIndex.txt'):
        with open(f'{saved_folder}/currentIndex.txt', 'a+') as newIndexFile:
            newIndexFile.write(str(0))
    with open(f'{saved_folder}/currentIndex.txt', 'r') as indexFile: 
        currentIndex = int(indexFile.read())
    return currentIndex


def saveDescriptors(descriptor_list):
    outfile = open(f'{saved_folder}/descriptor_list.obj', 'wb') 
    pickle.dump(descriptor_list, outfile)


def readDescriptors(max=100000):

    if os.path.exists(f'{saved_folder}/descriptor_list.obj'):
        infile = open(f'{saved_folder}/descriptor_list.obj', 'rb')
        desc_list = pickle.load(infile)
        if len(desc_list) < max:
            return desc_list
        else:
            return desc_list[:max]
    else:
        return []


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

def saveLoopClosures(lcc):
    lcpast = [lc[0] for lc in lcc]
    lcnow =  [lc[1] for lc in lcc]
    with open(f'{saved_folder}/lcpast.txt', 'w') as lcpFile:
        lcpFile.write(str(lcpast))
    with open(f'{saved_folder}/lcnow.txt', 'w') as lcnFile:
        lcnFile.write(str(lcnow))
    shutil.copyfile(f'{saved_folder}/lcpast.txt', '/root/ORB_SLAM2/Examples/Monocular/lcPast.txt')
    shutil.copyfile(f'{saved_folder}/lcnow.txt', '/root/ORB_SLAM2/Examples/Monocular/lcNow.txt')
    

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

def copyKFT():
    i = 1
    while os.path.exists(f'/root/ORB_SLAM2/Examples/Monocular/before{i}.txt'):
        shutil.copyfile(f'/root/ORB_SLAM2/Examples/Monocular/before{i}.txt', f'{saved_folder}/before{i}.txt')
        i+=1
    shutil.copyfile(f'/root/ORB_SLAM2/Examples/Monocular/after.txt', f'{saved_folder}/after.txt')

    return i

def readKFT(filename):
    global origin, theta, scale
    xs = []
    zs = []
    with open(f'{saved_folder}/{filename}.txt', 'r') as kftFile:
        lines = kftFile.readlines()
        data1 = lines[0].split(" ")
        point1 = ((-float(data1[1])*scale), (float(data1[3]))*scale)
        for line in lines:
            data = line.split(" ")
            point = ((-float(data[1]))*scale, (float(data[3]))*scale)
            x, z = rotate(point1, point, theta)
            xs.append(x+origin[0]-point1[0])
            zs.append(z+origin[1]-point1[1])
    return xs, zs

def readGT():
    xs = []
    zs = []
    with open(f'{sequence_folder}/groundtruth.txt', 'r') as kftFile:
        lines = kftFile.readlines()
        for i in range(len(lines)):
            if i > 2:
                data = lines[i].split(" ")
                xs.append(float(data[1]))
                zs.append(float(data[3]))
    return xs, zs


def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def showKeyPoints(image, keypoints):
    name = 'keypoints'

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 640*2, 480)

    if not os.path.exists(f'{name}.png'):
        for keypoint in keypoints:
            cv2.circle(image, (int(keypoint.pt[0]), int(keypoint.pt[1])), radius=1, color=(0,255,0), thickness=2)
    
    cv2.imshow(name, image)
    while True:
        if(cv2.waitKey(0) & 0xFF == ord('q')):
            break

    cv2.imwrite(f'{name}.png', image)