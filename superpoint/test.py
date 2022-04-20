import extractor
import common.handler as handler
import argparse
from pathlib import Path
import os
import cv2
import time


def runSuperpoint(name, targetframe, i=0):
    handler.readFolder(name)
    filenames, frames, last = handler.getNewFrames(first=targetframe, last=targetframe+1)
    start_time = time.time()
    extractor.runSuperpoint(frames)
    elapsed_time = time.time()-start_time
    print(f'Took {elapsed_time}')
    keypoints = extractor.all_keypoints[i]
    handler.showKeyPoints(frames[0], keypoints, save=True, new=True)

if __name__ == "__main__":
    extractor.start()
    # runSuperpoint('/root/MONO_LONG/', 1450)
    # runSuperpoint('/root/KITTI_06/', 26, i=0)
    # runSuperpoint('/root/KITTI_06/', 857, i=1)
    # runSuperpoint('/root/KITTI_06/', 171, i=2)
    runSuperpoint('/root/KITTI_06/', 997, i=0)



    
    # runSuperpoint('/root/MONO_LONG/', 2200)
    # runSuperpoint('/root/ORB_FR1/', 30)


    # for i in range(10):
    #     targetframe = ((i+40)*50)
    #     runSuperpoint('/root/MONO_LONG/', targetframe, i)
