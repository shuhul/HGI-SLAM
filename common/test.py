import common.handler as handler
import os


def runBoth(name, targetframe, i=0):
    handler.readFolder(name)
    filenames, frames, last = handler.getNewFrames(first=targetframe, last=targetframe+1)
    os.chdir('/root/HGI_SLAM/superpoint')
    suppoints = handler.readKPs()
    os.chdir('/root/HGI_SLAM/salgan')
    salpoints = handler.readKPs()
    handler.showKeyPointsBoth(frames[0], suppoints, salpoints)


if __name__ == "__main__":
    runBoth('/root/MONO_LONG/', 1450)