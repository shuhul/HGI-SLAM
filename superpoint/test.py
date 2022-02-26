import extractor
import common.handler as handler
import argparse
from pathlib import Path
import os
import cv2


if __name__ == "__main__":
    extractor.start()
    if not os.path.exists('keypoints.png'):
        handler.readFolder('/root/ORB_FR1/')
        filenames, frames, last = handler.getNewFrames(last=4)
        extractor.runSuperpoint(frames)
        keypoints = extractor.all_keypoints[0]
    else:
        keypoints = []
        frames = [cv2.imread('keypoints.png')]
    handler.showKeyPoints(frames[0], keypoints)
