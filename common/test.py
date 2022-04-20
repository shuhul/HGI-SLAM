import common.handler as handler
import os
import cv2
import math
import random


def runBoth(name, targetframe, i=0):
    handler.readFolder(name)
    filenames, frames, last = handler.getNewFrames(first=targetframe, last=targetframe+1)
    os.chdir('/root/HGI_SLAM/superpoint')
    suppoints = handler.readKPs()
    os.chdir('/root/HGI_SLAM/salgan')
    salpoints = handler.readKPs()
    handler.showKeyPointsBoth(frames[0], suppoints, salpoints)

def runOrb(name, targetframe):
    handler.readFolder(name)
    filenames, frames, last = handler.getNewFrames(first=targetframe, last=targetframe+1)
    orb = cv2.ORB_create(nfeatures=3000)
    image = frames[0]
    kps, des = orb.detectAndCompute(image, None)
    rkps = []
    for kp in kps:
        fail = False
        for rkp in rkps:
            dis = math.sqrt((rkp.pt[0] - kp.pt[0])**2 + (rkp.pt[1] - kp.pt[1])**2)
            if dis < random.randint(8,15):
                fail=True
        if not fail:
            rkps.append(kp)
    handler.showKeyPoints(image, rkps, save=False, new=True)



if __name__ == "__main__":
    runOrb('/root/KITTI_06/', 997)
    # runBoth('/root/MONO_LONG/', 1450)
    # runBoth('/root/KITTI_06/', 26)
    # runBoth('/root/KITTI_06/', 857)
    # runBoth('/root/KITTI_06/', 171)
    # runBoth('/root/KITTI_06/', 997)