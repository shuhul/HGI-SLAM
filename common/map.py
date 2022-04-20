import handler

if __name__ == "__main__":
    folder = '/root/KITTI_06' #00 02 05 06 07 09
    handler.readFolder(folder)
    handler.showMap()

    lcc = [(26,857), (171, 997)]
    handler.showLoopClosurePairs(lcc)

    # handler.showVideo(skip=10)