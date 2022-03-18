import extractor
import common.handler as handler
import argparse
from pathlib import Path
import os
import cv2
import time

if __name__ == "__main__":
    extractor.start()

    handler.readFolder('/root/ORB_FR1/')
    filenames, frames, last = handler.getNewFrames(last=100)
    start_time = time.time()
    extractor.runSuperpoint(frames)
    elapsed_time = time.time()-start_time
    print(f'Took {elapsed_time}')
    keypoints = extractor.all_keypoints[0]
    # print(keypoints)


    # if not os.path.exists('keypoints.png'):
    #     handler.readFolder('/root/ORB_FR1/')
    #     filenames, frames, last = handler.getNewFrames(last=4)
    #     extractor.runSuperpoint(frames)
    #     keypoints = extractor.all_keypoints[0]
    # else:
    #     keypoints = []
    #     frames = [cv2.imread('keypoints.png')]
    handler.showKeyPoints(frames[0], keypoints, new=True)
