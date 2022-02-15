import os
import numpy as np
from tqdm import tqdm
import cv2
import glob
from utils import *
from constants import *
from models.model_bce import ModelBCE
import common.handler as handler
import argparse
import theano


def test(path_to_images, path_output_maps, model_to_test=None):
    list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(path_to_images, '*'))]
    # Load Data
    list_img_files.sort()
    for curr_file in tqdm(list_img_files, ncols=20):
        print(os.path.join(path_to_images, curr_file + '.jpg'))
        img = cv2.cvtColor(cv2.imread(os.path.join(path_to_images, curr_file + '.jpg'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        predict(model=model_to_test, image_stimuli=img, name=curr_file, path_output_maps=path_output_maps)

def preprocess_image(image, img_size):
    img = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)
    img_orig = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_preprocessed = img

    return img_preprocessed, img_orig

def generateHeatmap(img):
    model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1], batch_size=8)
    load_weights(model.net['output'], path='gen_', epochtoload=90)
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

def generateKeypoints(heatmap):
    pass
def generateDescriptor(keypoints):
    pass

def runSalgan(frames):
    if len(frames) == 0:
        print(f'Skipping {handler.readCurrentIndex()} already processed frames')
        return []
    descriptor_list = []

    count = 1
    for frame in frames:
        print(f'Proccessing frame {count} of {len(frames)}')
        img, img_orig = preprocess_image(frame, (640, 480))
        # heatmap = generateHeatmap(img)
        heatmap = cv2.imread('heat.png')


        # cv2.imwrite('orig.png', img_orig)
        # cv2.imwrite('heat.png', heatmap)
        
        # keypoints = generateKeypoints(heatmap)
        # descriptor = generateDescriptor(keypoints)
        # descriptor_list.append(descriptor)
        count += 1
    return descriptor_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Run superpoint on a image sequence')
    parser.add_argument('path_to_sequence', type=str)
    args = parser.parse_args()

    sequence_folder = args.path_to_sequence


    # Create network
    # model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1], batch_size=8)
    # # Here need to specify the epoch of model sanpshot
    # load_weights(model.net['output'], path='gen_', epochtoload=90)
    # # Here need to specify the path to images and output path
    # test(path_to_images='../images/', path_output_maps='../saliency/', model_to_test=model)


    saved_folder = 'saved'

    print('\n-------Generating Descriptors--------\n')

    handler.readFolder(sequence_folder, saved_folder)

    num_frames = 1

    filenames, new_frames = handler.getNewFrames(last=num_frames, shouldsave=False)

    runSalgan(new_frames)

    # descriptor_list = handler.readDescriptors() + featureExtractor(new_frames)

    # handler.saveDescriptors(descriptor_list)

