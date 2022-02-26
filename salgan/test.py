import predictor
import common.handler as handler
import argparse
from pathlib import Path
import os
import cv2


if __name__ == "__main__":

    predictor.start()
    if not os.path.exists('keypoints.png'):
        handler.readFolder('/root/ORB_FR1/')
        filenames, frames, last = handler.getNewFrames(last=4)
        predictor.runSalgan(frames)
        keypoints = predictor.all_keypoints[0]
    else:
        keypoints = []
        frames = [cv2.imread('keypoints.png')]
    handler.showKeyPoints(frames[0], keypoints)
