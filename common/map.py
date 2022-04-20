import handler

if __name__ == "__main__":
    folder = '/root/KITTI_02'
    handler.readFolder(folder)
    handler.showMap()
    # handler.showVideo(skip=10)