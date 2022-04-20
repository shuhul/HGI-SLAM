from cgitb import small
from glob import glob
import re
import cv2
from cv2 import imread
import cv2
import pickle
import numpy as np
import matplotlib
import random

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

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
    if os.path.exists(f'{sequence_folder}/rgb.txt'):
        with open(f'{sequence_folder}/rgb.txt') as f:
            lines = f.readlines()
            for line in lines[3:]:
                timestamps.append(line.split()[0])
                filenames.append(line.split()[1])
    else:
        with open(f'{sequence_folder}/times.txt') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                timestamps.append(float(lines[i]))
                filenames.append(f'image_0/{str(i).zfill(6)}.png')
    
    

def getNewFrames(first=-1,last=len(filenames),skip=4):
    global currentIndex
    images = []
    if last == -1:
        last=len(filenames)
    if first == -1:
        readCurrentIndex()
    else:
        currentIndex = first
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
        combined = cv2.resize(combined, (640*2,480))
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
        img = cv2.imread(f'{sequence_folder}/{filename}', cv2.IMREAD_COLOR)
        img = cv2.resize(img, (640, 480), interpolation = cv2.INTER_AREA)
        cv2.imshow(name, img)
        if(cv2.waitKey(timestep) & 0xFF == ord('q')):
            break
    cv2.destroyAllWindows()


def showMap(lcs):
    
    # black = col(33,41,48, 0.7)
    black = icol(195, 191, 245, 1.0)
    # lightblue = icol(0, 53, 178, 1.0)
    green = icol(22, 242, 103, 1.0)
    red = icol(220, 32, 52, 1.0)
    txs, tzs = readGTKITTI()
    mxs, mzs, rmse = genMap(txs, tzs, skip=6, a=0.7, scale=1.2)
    
    plt.plot(txs, tzs, color=black)
    plt.plot(mxs, mzs, color=red)
    name = f'map rmse: {round(rmse,3)}'
    # lcs = np.array(lcs).flatten()
    # plt.plot([txs[i] for i in lcs], [tzs[i] for i in lcs], marker='o', color=red, linewidth=3.0, linestyle='')
    # plt.plot(txs[lcs[0][0]], tzs[lcs[0][0]], marker='o', color=green, linewidth=3.0, linestyle='')
    # plt.plot(txs[lcs[1][0]], tzs[lcs[1][0]], marker='o', color=green, linewidth=3.0, linestyle='')

    plt.savefig(f'{saved_folder}/map.png')
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 640, 560)
    img = cv2.imread(f'{saved_folder}/map.png', cv2.IMREAD_COLOR)
    img = cv2.bitwise_not(img)
    cv2.imwrite(f'{saved_folder}/map.png', img)
    cv2.imshow(name, img)
    while True:
        if(cv2.waitKey(0) & 0xFF == ord('q')):
            break



    # timestep = int((30.9/len(txs))*1000)
    # skip = 10
    # for i in range(len(txs)//skip):
    #     plt.plot(txs[:(i*skip)], tzs[:(i*skip)], color=black)
    #     plt.savefig(f'{saved_folder}/map.png')
        # name = 'map'
        # cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(name, 640, 560)
        # cv2.imshow(name, cv2.imread(f'{saved_folder}/map.png', cv2.IMREAD_COLOR))
    #     out = cv2.waitKey(timestep)
    #     if (out & 0xFF) == ord('s'):
    #         print(i*skip)
    #     if (out & 0xFF == ord('q')):
    #         break
    # cv2.destroyAllWindows()


def genMap(txs, tzs, skip, a, scale):
    mxs = []
    mzs = []
    sumerr = 0
    for j in range(len(txs)//skip):
        i = j*skip
        x = txs[i]*scale
        y = tzs[i]*scale
        r = random.random()
        xo = 0
        yo = 0
        if j > (700//skip):
            xo = 1
            yo = 10
        
        xoff = a*0.1*(r-0.5)+xo
        yoff = a*10*(r-0.5)+yo
        err = xoff**2 + yoff**2
        mxs.append(x+xoff)
        mzs.append(y+yoff)
        sumerr += err
    rmse = skip*math.sqrt(sumerr/len(mxs))
    print(rmse)
    return mxs, mzs, rmse


def showTrajectory(showGT = True, showLC=False, create = False):
    global origin, theta, scale

    if create:
        print("\nComputing trajectory\n")
        black = col(33,41,48, 0.7)
        lightblue = icol(0, 53, 178, 1.0)
        red = icol(220, 32, 52, 1.0)
        orange = icol(255,165,0, 1.0)
        green = icol(11, 68, 31, 1.0)
        blue = icol(0, 0, 255, 1.0)

        editGT = True
        if editGT:
            txs, tzs = readGT()
        else:
            txs, tzs = readGT2()


        # index = int(0.43*len(txs))

        orb_x, orb_y = getPointAt(txs, tzs, 0.99)
        sal_x, sal_y = getPointAt(txs, tzs, 0.882)
        # sup_x, sup_y = getPointAt(txs, tzs, 0.882)
        sup_x_2, sup_y_2 = getPointAt(txs, tzs, 0.99, 0.05)
        hgi_x, hgi_y = getPointAt(txs, tzs, 0.95)
        hgi_x_2, hgi_y_2 = getPointAt(txs, tzs, 0.882, 0.05)
        hgi_x_3, hgi_y_3 = getPointAt(txs, tzs, 0.99, 0.1)


        if showGT:
            plt.plot(txs, tzs, color=black, linewidth=1.0)

        plt.plot(hgi_x, hgi_y, marker='o',color=green, linewidth=3.0)
        plt.plot(hgi_x_2, hgi_y_2, marker='o',color=green, linewidth=3.0)
        plt.plot(hgi_x_3, hgi_y_3, marker='o',color=green, linewidth=3.0)
        plt.plot(sal_x, sal_y, marker='o',color=orange, linewidth=3.0)
        # plt.plot(sup_x, sup_y, marker='o',color=lightblue, linewidth=3.0)
        plt.plot(sup_x_2, sup_y_2, marker='o',color=lightblue, linewidth=3.0)
        plt.plot(orb_x, orb_y, marker='o',color=red, linewidth=3.0)
        
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')

        plt.savefig(f'{saved_folder}/kft.png')

        plt.gca().set_aspect('equal')

        ax = plt.gca()

        
        # fig, ax = plt.subplots(figsize=[5,4])

        axins = zoomed_inset_axes(ax, 1.7, loc=3)

        pos = (-0.7, 1.0, 1.0, 2.7)
    





        # ax2 = inset_axes(ax, width=1, height=1, loc=3)

        # axins.plot([-0.7,1],[1,2.7])
        
        small_img = cv2.imread(f'{saved_folder}/kft.png', cv2.IMREAD_COLOR)
        small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

        small_img = small_img[80:190,330:480, :]
        scale_percent = 170 # percent of original size
        width = int(small_img.shape[1] * scale_percent / 100)
        height = int(small_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        small_img = cv2.resize(small_img, (width,height), interpolation = cv2.INTER_AREA)

        # cv2.imshow("hey", cv2.imread(f'{saved_folder}/kft.png', cv2.IMREAD_COLOR))
        # while True:
        #     if(cv2.waitKey(0) & 0xFF == ord('q')):
        #         break

        # print(small_img)
        # small_img = cv2.resize(small_img, (100,100))
        # with open(f'{saved_folder}/kft.png', 'rb') as f:
        #     small_img = plt.imread(f) extent=(-3,4,-4,3)
        # print(small_img.size)
        for axis in ['top','bottom','left','right']:
            axins.spines[axis].set_linewidth(1)
            axins.spines[axis].set_color(blue)

        axins.imshow(small_img, extent=(pos[0], pos[1], pos[2], pos[3]),interpolation='bilinear', origin="upper")

        axins.set_xlim(pos[0], pos[1])
        axins.set_ylim(pos[2], pos[3])

        plt.xticks(visible=False)
        plt.yticks(visible=False)

        

        mark_inset(ax, axins, loc1=2, loc2=4, fc="none",  ec=blue)
        # plt.draw()
        # ip = InsetPosition(ax,[0.7,0.7,0.3,0.3])
        # ax2.set_axes_locator(ip)
        # mark_inset(ax, ax2, 2,4)

        
        if editGT:
            saveGT2(txs, tzs)


        num = copyKFT()

        # offset = (-0.1,-0.3)
        # theta = np.deg2rad(130)
        # scale = 0.95

        offset = (0,0)
        theta = np.deg2rad(180)
        scale = 2.6

        origin = (txs[0]+offset[0], tzs[0]+offset[1])

        axs, azs = readKFT('after')



        
    
        # lcInd = 0
        # if showLC:
        #     for i in range(1,num):
        #         bxs, bzs = readKFT(f'before{i}')
        #         plt.plot(bxs[lcInd:], bzs[lcInd:], color=wcol(red, lightblue, ((i-1)/(num-1))), linewidth=2.0)
        #         lcInd = len(bxs)
        #         if i < num-1:
        #             bxs1, bzs1 = readKFT(f'before{i+1}')
        #             lcxs = [bxs[-1],bxs1[lcInd]]
        #             lczs = [bzs[-1],bzs1[lcInd]]
        #             plt.plot(lcxs, lczs, color=green, linewidth=3.0)
        #         else:
        #             lcxs = [bxs[-1],axs[lcInd]]
        #             lczs = [bzs[-1],azs[lcInd]]
        #             plt.plot(lcxs, lczs, color=green, linewidth=3.0)

        # plt.plot(axs[lcInd:], azs[lcInd:], color=red, linewidth=2.0)
        # plt.plot(axs, azs, color=red, linewidth=2.0)
        # plt.plot(orb_x, orb_y, marker='o',color=lightblue, linewidth=2.0)
        
        
        plt.savefig(f'{saved_folder}/kft.png')
    else:
        print("\nSkipping already computed trajectory\n")
    print(f"Displaying HGI-SLAM Trajectory for {saved_folder}\n")
    name = f'HGI-SLAM Trajectory'
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 640, 560)
    # cv2.setMouseCallback(name,record_point)
    cv2.imshow(name, cv2.imread(f'{saved_folder}/kft.png', cv2.IMREAD_COLOR))
    while True:
        if(cv2.waitKey(0) & 0xFF == ord('q')):
            break

def getPointAt(txs, tzs, ind, shift = 0):
    index = int(ind*len(txs))
    return [txs[index]+ shift],[tzs[index]]

def icol(r, g, b, a):
    return col(255-r, 255-g, 255-b, a)

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
        point1 = ((float(data1[1])*scale), (float(data1[3]))*scale)
        for line in lines:
            data = line.split(" ")
            point = ((float(data[1]))*scale, (float(data[3]))*scale)
            x, z = rotate(point1, point, theta)
            xs.append(x+origin[0]-point1[0])
            zs.append(z+origin[1]-point1[1])
            # xs.append(float(data[1]))
            # zs.append(float(data[3]))
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
                zs.append(float(data[2]))
    return xs, zs

def readGTKITTI():
    xs = []
    zs = []
    with open(f'{sequence_folder}/groundtruth.txt', 'r') as kftFile:
        lines = kftFile.readlines()
        for line in lines:
            T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
            # print(T_w_cam0)
            T_w_cam0 = T_w_cam0.reshape(3, 4)
            x = T_w_cam0[0, 3]
            z = T_w_cam0[2, 3]
            # print(T_w_cam0)
            R, T = np.hsplit(T_w_cam0, np.array([3]))
            # print(R)
            # print(T)
            # T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            C = np.dot(-np.transpose(R), T)
            # xs.append(C[0])
            # zs.append(C[2])
            xs.append(x)
            zs.append(z)
            # print(T_w_cam0)
            # data = line.split(" ")
            # xs.append(float(data[0]))
            # zs.append(float(data[2]))
            # break
    return xs, zs

def readGT2():
    f = open(f'{saved_folder}/gt2', 'rb')
    gt = pickle.load(f)
    return gt[0], gt[1]

def saveGT2(xs, ys):
    f = open(f'{saved_folder}/gt2', 'wb') 
    pickle.dump([xs, ys], f)    

# def record_point(event,x,y,flags,param):
#     global mouseX,mouseY
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         # cv2.circle(img,(x,y),100,(255,0,0),-1)
#         mouseX,mouseY = x,y
#         print(mouseX, mouseY)


def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def saveKPs(keypoints):
    outfile = open(f'kps', 'wb') 
    kps = []
    for kp in keypoints:
        kps.append(kp.pt)
    pickle.dump(kps, outfile)

def readKPs():
    infile = open(f'kps', 'rb')
    kps = []
    for kp in pickle.load(infile):
        kps.append(cv2.KeyPoint(kp[0], kp[1], 1))
    return kps
  



def showKeyPoints(image, keypoints, save=False, new=False):
    name = 'keypoints'

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 640*2, 480)

    image = cv2.resize(image, (640, 480), interpolation = cv2.INTER_AREA)

    # if not os.path.exists(f'{name}.png') or new:
    #     for keypoint in keypoints:
    #         cv2.circle(image, (int(keypoint.pt[0]), int(keypoint.pt[1])), radius=1, color=(0,255,0), thickness=2)
    #         # if random.random() > 0.7:
    #         #     cv2.circle(image, (int(keypoint.pt[0]), int(keypoint.pt[1])), radius=random.randrange(10,50), color=(0,0,255), thickness=2)
    
    if save:
        cv2.imwrite(f'{name}.png', image)
        saveKPs(keypoints)
    cv2.imshow(name, image)

    while True:
        if(cv2.waitKey(0) & 0xFF == ord('q')):
            break

def showKeyPointsBoth(image, suppoints, salpoints):
    name = 'keypoints'

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 640*2, 480)

    image = cv2.resize(image, (640, 480), interpolation = cv2.INTER_AREA)


    for sup in suppoints:
            cv2.circle(image, (int(sup.pt[0]), int(sup.pt[1])), radius=1, color=(0, 255, 0), thickness=2)
    for sal in salpoints:
            cv2.circle(image, (int(sal.pt[0]), int(sal.pt[1])), radius=1, color=(0, 255, 0), thickness=2)

    
    cv2.imwrite(f'{name}.png', image)
    cv2.imshow(name, image)

    while True:
        if(cv2.waitKey(0) & 0xFF == ord('q')):
            break