import extractor
import common.handler as handler
import argparse
from pathlib import Path
import os
import cv2
import time
import math
import random
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import common.bagofwords as bow
from sklearn.cluster import KMeans


def runSuperpoint(name, targetframe, i=0):
    handler.readFolder(name)
    filenames, frames, last = handler.getNewFrames(first=targetframe, last=targetframe+1)


    # sup_desc = handler.readDescriptors(max=targetframe)

    # desc = np.array(sup_desc[-1])

    # pca = PCA(n_components=3)
    # pc = pca.fit_transform(desc)

    # pc = pc[::2]

    # xs = [c[0] for c in pc]
    # ys = [c[1] for c in pc]
    # zs = [c[2] for c in pc]

    # plt.plot(xs, ys, 'ro')

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # # ax.set_title('Descriptor Components')
    # ax.set_xlabel('Component 1')
    # ax.set_ylabel('Component 2')
    # ax.set_zlabel('Component 3')
    # ax.scatter3D(xs, ys, zs, c=zs, cmap='Oranges')

    # plt.savefig('temp.png')
    # img = cv2.imread('temp.png', cv2.IMREAD_COLOR)
    # img = cv2.bitwise_not(img)
    # cv2.imwrite('temp.png', img)


    # print(np.array(desc).shape)
    
    # start_time = time.time()
    # extractor.runSuperpoint(frames)
    # elapsed_time = time.time()-start_time
    # print(f'Took {elapsed_time}')
    # keypoints = extractor.all_keypoints[i]
    # rkps = keypoints
    # rkps = []
    # for kp in keypoints:
    #     fail = False
    #     for rkp in rkps:
    #         dis = math.sqrt((rkp.pt[0] - kp.pt[0])**2 + (rkp.pt[1] - kp.pt[1])**2)
    #         if dis < random.randrange(15, 30):
    #             fail=True
    #     if not fail:
    #         rkps.append(kp)
    # handler.showKeyPoints(frames[0], rkps, save=True, new=True)

if __name__ == "__main__":
    handler.readFolder('/root/MONO_LONG/') 
    sup_desc = handler.readDescriptors(max=2203)

    pca = PCA(n_components=2)
    df = pca.fit_transform(sup_desc[0])

    kmeans = KMeans(n_clusters=8)

    label = kmeans.fit_predict(df)

    u_labels = np.unique(label)

    for i in u_labels:
        plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i, cmap='inferno')
    plt.legend()
    plt.savefig('bow.png')

    img = cv2.imread('bow.png', cv2.IMREAD_COLOR)
    img = cv2.bitwise_not(img)
    cv2.imwrite('bow.png', img)




    # extractor.start()
    # runSuperpoint('/root/MONO_LONG/', 2203)
    # runSuperpoint('/root/KITTI_06/', 26, i=0)
    # runSuperpoint('/root/KITTI_06/', 857, i=1)
    # runSuperpoint('/root/KITTI_06/', 171, i=2)
    # runSuperpoint('/root/KITTI_06/', 997, i=0)



    
    # runSuperpoint('/root/MONO_LONG/', 2200)
    # runSuperpoint('/root/ORB_FR1/', 30)


    # for i in range(10):
    #     targetframe = ((i+40)*50)
    #     runSuperpoint('/root/MONO_LONG/', targetframe, i)
