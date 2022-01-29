import cv2

timestamps = []
filenames = []
images = []
sequence_folder = ''

def readFolder(folder):
    global sequence_folder
    sequence_folder = folder
    with open(f'{sequence_folder}/rgb.txt') as f:
        lines = f.readlines()
        for line in lines[3:]:
            timestamps.append(line.split()[0])
            filenames.append(line.split()[1])

def getFrames(n=len(filenames)):
    for filename in filenames[:n]:
        images.append(cv2.imread(f'{sequence_folder}/{filename}'))
    return images


def showVideo():
    timestep = int((30.9/len(filenames))*1000)
    for filename in filenames:
        cv2.imshow('Video Test', cv2.imread(f'{sequence_folder}/{filename}'))
        if(cv2.waitKey(timestep) & 0xFF == ord('q')):
            break