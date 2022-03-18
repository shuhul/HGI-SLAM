import re
from cv2 import imread
import matplotlib
import os,sys

# sys.stderr = open(os.devnull, 'w')
import tensorflow as tf
# sys.stdout = sys.__stderr__

import argparse
from pathlib import Path

import cv2
import numpy as np

import common.handler as handler

import common.bowhandler as bowh

import pickle 


from superpoint.settings import EXPER_PATH

weights_dir = ""
sequence_folder = ""
num = 0
all_keypoints = []
train = ""
img_size = (0,0)
keep_k_best = 0
scale = 3

def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map,
                                                 keep_k_points=100):

    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    # Extract keypoints
    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)

    keypoints = select_k_best(keypoints, keep_k_points)
    keypoints = keypoints.astype(int)

    # Get descriptors for keypoints
    desc = descriptor_map[keypoints[:, 0], keypoints[:, 1]]

    # Convert from just pts to cv2.KeyPoints
    keypoints = [cv2.KeyPoint(p[1]*scale, p[0]*scale, 1) for p in keypoints]

    return keypoints, desc


def preprocess_image(image, img_size):
    img = cv2.resize(image, img_size)
    img_orig = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.

    return img_preprocessed, img_orig

def runSuperpoint(frames):
    global all_keypoints
    if len(frames) == 0:
        return []
    descriptor_list = []
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], str(weights_dir))
        input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
        output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
        output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

        count = 1
        for frame in frames:
            print(f'Proccessing frame {count} of {len(frames)}')
            img, img_orig = preprocess_image(frame, img_size)
            out = sess.run([output_prob_nms_tensor, output_desc_tensors], feed_dict={input_img_tensor: np.expand_dims(img, 0)})
            keypoint_map = np.squeeze(out[0])
            descriptor_map = np.squeeze(out[1])
            keypoints, descriptor = extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map, keep_k_best)
            all_keypoints.append(keypoints)
            descriptor_list.append(descriptor)
            count += 1
        return descriptor_list


def start():
    global weights_dir, sequence_folder, num, train, img_size, keep_k_best

    weights_name = "sp_v6"
    img_size = (640//scale, 480//scale)
    keep_k_best = 1000

    
    weights_root_dir = Path(EXPER_PATH, 'saved_models')
    weights_root_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(weights_root_dir, weights_name)

    os.chdir("/root/HGI_SLAM/superpoint")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Run superpoint on a image sequence')
    parser.add_argument('path_to_sequence', type=str)
    parser.add_argument('num_imgs', type=int)
    parser.add_argument('training', type=str)
    args = parser.parse_args()

    sequence_folder = args.path_to_sequence
    num = args.num_imgs
    train = True if args.training == "y" else False

    start()

    print("\nRunning Superpoint")

    bowh.run(sequence_folder, runSuperpoint, max_frame=num, training=train, detecting=True, max_distance=0.1)
