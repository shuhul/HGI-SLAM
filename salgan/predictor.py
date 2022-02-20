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


def preprocess_image(image, img_size):
    img = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)
    img_orig = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_preprocessed = img
    img_gs = cv2.cvtColor(img_preprocessed, cv2.COLOR_RGB2GRAY)
    return img_preprocessed, img_gs, img_orig

def extractHeatmap(img, model):
    size = (img.shape[1], img.shape[0])
    blur_size = 5

    if img.shape[:2] != (model.inputHeight, model.inputWidth):
        img = cv2.resize(img, (model.inputWidth, model.inputHeight), interpolation=cv2.INTER_AREA)

    blob = np.zeros((1, 3, model.inputHeight, model.inputWidth), theano.config.floatX)

    blob[0, ...] = (img.astype(theano.config.floatX).transpose(2, 0, 1))

    result = np.squeeze(model.predictFunction(blob))
    saliency_map = (result * 255).astype(np.uint8)
    saliency_map = cv2.resize(saliency_map, size, interpolation=cv2.INTER_CUBIC)
    saliency_map = cv2.GaussianBlur(saliency_map, (blur_size, blur_size), 0)
    saliency_map = np.clip(saliency_map, 0, 255)

    return saliency_map


def runSalgan(frames):
    if len(frames) == 0:
        print(f'Skipping {handler.readCurrentIndex()} already processed frames')
        return []
    descriptor_list = []

    model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1], batch_size=8)
    load_weights(model.net['output'], path='gen_', epochtoload=90)

    count = 1
    for frame in frames:
        print(f'Proccessing frame {count} of {len(frames)}')
        img, img_gs, img_orig = preprocess_image(frame, (640, 480))
        heatmap = extractHeatmap(img, model)
        # heatmap = cv2.imread('heat.png', IMREAD_GRAYSCALE)
        # keypoints = generator.generateKeypoints(img_gs, heatmap)
        # pickle.dump(keypoints, open("keypoints", "wb"))
        # keypoints = pickle.load(open("keypoints", "rb"))
        # keypoints = generator.kpsToKPS(keypoints)
        # descriptor = generator.generateDescriptors(img_gs, keypoints)

        keypoints, descriptor = generator.generateKeypointsAndDescriptors(img_gs, heatmap)

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


        # cv2.imwrite('orig.png', img_orig)
        # cv2.imwrite('heat.png', heatmap)
        
        
        descriptor_list.append(descriptor)
        count += 1
    return descriptor_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Run superpoint on a image sequence')
    parser.add_argument('path_to_sequence', type=str)
    args = parser.parse_args()

    sequence_folder = args.path_to_sequence
    
    bowh.run(sequence_folder, runSalgan, num_frames=300, training=False, detecting=True, max_distance=0.5)
