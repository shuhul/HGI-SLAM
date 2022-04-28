import handler

if __name__ == "__main__":
    # folder = '/root/KITTI_06' #00 02 05 06 07 09
    folder = '/root/MONO_DESK'
    handler.readFolder(folder)
    # lcc = [(26,857), (171, 997)]
    lcc = [(14,51)]
    handler.showMap(lcc, isfr=True)
    # handler.showLoopClosurePairs(lcc)

    # handler.showVideo(skip=10)