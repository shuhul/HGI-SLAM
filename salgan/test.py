import predictor
import common.handler as handler
import argparse
from pathlib import Path
import os
import cv2
import time



if __name__ == "__main__":

    predictor.start()
    handler.readFolder('/root/ORB_FR1/')
    filenames, frames, last = handler.getNewFrames(last=1)

    predictor.loadModel()
    start_time = time.time()
    predictor.runSalgan(frames)
    elapsed_time = time.time()-start_time
    
    print(f'Took {elapsed_time}')
    keypoints = predictor.all_keypoints[0]
    # if not os.path.exists('keypoints.png'):
    #     handler.readFolder('/root/ORB_FR1/')
    #     filenames, frames, last = handler.getNewFrames(last=4)
    #     predictor.runSalgan(frames)
    #     keypoints = predictor.all_keypoints[0]
    # else:
    #     keypoints = []
    #     frames = [cv2.imread('keypoints.png')]
    handler.showKeyPoints(frames[0], keypoints, save=True, new=True)
