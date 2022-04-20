import predictor
import common.handler as handler
import argparse
from pathlib import Path
import os
import cv2
import time


def runSalgan(name, targetframe, i=0):
    handler.readFolder(name)
    filenames, frames, last = handler.getNewFrames(first=targetframe, last=targetframe+1)
    predictor.loadModel()
    start_time = time.time()
    predictor.runSalgan(frames)
    elapsed_time = time.time()-start_time
    print(f'Took {elapsed_time}')
    keypoints = predictor.all_keypoints[i]
    handler.showKeyPoints(frames[0], keypoints, save=True, new=True)

if __name__ == "__main__":

    predictor.start()
    runSalgan('/root/MONO_LONG/', 2200)
    # runSalgan('/root/MONO_LONG/', 1450)
    # runSalgan('/root/KITTI_06/', 26, i=0)
    # runSalgan('/root/KITTI_06/', 857, i=0)
    # runSalgan('/root/KITTI_06/', 171, i=0)
    # runSalgan('/root/KITTI_06/', 997, i=0)


    # runSuperpoint('/root/MONO_LONG/', 2200)
    # runSuperpoint('/root/ORB_FR1/', 30)


    # for i in range(10):
    #     targetframe = ((i+40)*50)
    #     runSuperpoint('/root/MONO_LONG/', targetframe, i)