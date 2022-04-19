from copyreg import pickle
import os
from cv2 import IMREAD_GRAYSCALE
import numpy as np
from tqdm import tqdm
import cv2
import glob
from utils import *
from constants import *
from models.model_bce import ModelBCE
import common.handler as handler
import generator
import argparse
import theano
import pickle
import matplotlib.pyplot as plt
import common.bowhandler as bowh


scalex = 1 # 10
scaley = 1 # 10

all_keypoints = []
model = None

def preprocess_image(image, img_size):
    img = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)
    img_orig = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_preprocessed = img
    img_gs = cv2.cvtColor(img_preprocessed, cv2.COLOR_RGB2GRAY)
    return img_preprocessed, img_gs, img_orig

def extractHeatmap(img):
    global model
    size = (img.shape[1], img.shape[0])
    blur_size = 5

    if img.shape[:2] != (model.inputHeight, model.inputWidth):
        img = cv2.resize(img, (model.inputWidth, model.inputHeight), interpolation=cv2.INTER_AREA)
    
    cv2.imwrite('in.png', img)

    blob = np.zeros((1, 3, model.inputHeight, model.inputWidth), theano.config.floatX)

    blob[0, ...] = (img.astype(theano.config.floatX).transpose(2, 0, 1))

    result = np.squeeze(model.predictFunction(blob))
    saliency_map = (result * 255).astype(np.uint8)
    saliency_map = cv2.resize(saliency_map, size, interpolation=cv2.INTER_CUBIC)
    saliency_map = cv2.GaussianBlur(saliency_map, (blur_size, blur_size), 0)
    saliency_map = np.clip(saliency_map, 0, 255)

    return saliency_map


def loadModel():
    global model
    model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1], batch_size=8)
    # model = ModelBCE(640//scalex, 480//scaley, batch_size=8)
    load_weights(model.net['output'], path='gen_', epochtoload=90)



def runSalgan(frames):
    global all_keypoints, model, scalex, scaley

    if len(frames) == 0:
        # print(f'Skipping {handler.readCurrentIndex()} already processed frames')
        return []
    descriptor_list = []
   
    count = 1
    for frame in frames:
        print(f'Proccessing frame {count} of {len(frames)}')
        img, img_gs, img_orig = preprocess_image(frame, (640//scalex, 480//scaley))
        # cv2.imwrite('orig.png', img_orig)
        # cv2.imwrite('gray.png', img_gs)
        # cv2.imwrite('in.png', img)
        heatmap = extractHeatmap(img)
        cv2.imwrite('heat.png', heatmap)
        # heatmap = cv2.imread('heat.png', IMREAD_GRAYSCALE)
        # keypoints = generator.generateKeypoints(img_gs, heatmap)
        # pickle.dump(keypoints, open("keypoints", "wb"))
        # keypoints = pickle.load(open("keypoints", "rb"))
        # keypoints = generator.kpsToKPS(keypoints)
        # descriptor = generator.generateDescriptors(img_gs, keypoints)
        keypoints = []
        descriptor = []
        generator.scalex = scalex
        generator.scaley = scaley
        keypoints = generator.generateKeypoints(img_gs, heatmap)
        # keypoints, descriptor = generator.generateKeypointsAndDescriptors(img_gs, heatmap)
        all_keypoints.append(keypoints)

        # print("\nDescriptor from SALGAN\n")
        # print(descriptor[0])

        # sift = cv2.xfeatures2d.SIFT_create()
        # kp, des = sift.detectAndCompute(img_gs,None)

        # print("\nDescriptor from SIFT\n")
        # print(des[0])
        # print("\n\n")

        # for keypoint in keypoints:
        #     cv2.circle(img_orig, keypoint, 1, color=(0,255,0), thickness=2)
        # cv2.imwrite("key.png", img_orig)

        
        descriptor_list.append(descriptor)
        count += 1
    return descriptor_list


def start():
    os.chdir("/root/HGI_SLAM/salgan")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Run salgan on a image sequence')
    parser.add_argument('path_to_sequence', type=str)
    parser.add_argument('num_imgs', type=int)
    parser.add_argument('training', type=str)
    args = parser.parse_args()

    start()

    sequence_folder = args.path_to_sequence
    num = args.num_imgs
    train = True if args.training == "y" else False
    loadModel()
    bowh.run(sequence_folder, runSalgan, max_frame=num, training=train, num_clusters=8, num_neighbors=5, detecting=False, max_distance=1.2)
