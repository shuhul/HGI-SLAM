import matplotlib
import os,sys

sys.stderr = open(os.devnull, 'w')
import tensorflow as tf
sys.stdout = sys.__stderr__

import argparse
from pathlib import Path

import cv2
import numpy as np

import BagOfWords
import handler

from superpoint.settings import EXPER_PATH

def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map,
                                                 keep_k_points=1000):

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
    keypoints = [cv2.KeyPoint(p[1], p[0], 1) for p in keypoints]

    return keypoints, desc


def preprocess_image(img_file, img_size):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, img_size)
    img_orig = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.

    return img_preprocessed, img_orig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Run superpoint on a image sequence')
    parser.add_argument('weights_name', type=str)
    parser.add_argument('path_to_sequence', type=str)
    parser.add_argument('--H', type=int, default=480,
                        help='The height in pixels to resize the images to. \
                                (default: 480)')
    parser.add_argument('--W', type=int, default=640,
                        help='The width in pixels to resize the images to. \
                                (default: 640)')
    parser.add_argument('--k_best', type=int, default=1000,
                        help='Maximum number of keypoints to keep \
                        (default: 1000)')
    args = parser.parse_args()

    weights_name = args.weights_name
    sequence_folder = args.path_to_sequence
    img_size = (args.W, args.H)

    handler.readFolder(sequence_folder)


    frames = handler.getFrames(10)

    # print('hi')

    cv2.destroyAllWindows() 

    # cv2.waitKey(0)
    # plt.show()
    # print(imgs_folder)
    # keep_k_best = args.k_best
    # weights_root_dir = Path(EXPER_PATH, 'saved_models')
    # weights_root_dir.mkdir(parents=True, exist_ok=True)
    # weights_dir = Path(weights_root_dir, weights_name)

    # graph = tf.Graph()
    # with tf.Session(graph=graph) as sess:
    #     tf.saved_model.loader.load(sess,
    #                                [tf.saved_model.tag_constants.SERVING],
    #                                str(weights_dir))

    #     input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
    #     output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
    #     output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

    #     img1, img1_orig = preprocess_image(img1_file, img_size)
    #     out1 = sess.run([output_prob_nms_tensor, output_desc_tensors],
    #                     feed_dict={input_img_tensor: np.expand_dims(img1, 0)})
    #     keypoint_map1 = np.squeeze(out1[0])
    #     descriptor_map1 = np.squeeze(out1[1])
    #     kp1, desc1 = extract_superpoint_keypoints_and_descriptors(
    #             keypoint_map1, descriptor_map1, keep_k_best)

    #     print("----------------Start----------------")

    #     BagOfWords.runBoW(kp1, desc1)

        # img2, img2_orig = preprocess_image(img2_file, img_size)
        # out2 = sess.run([output_prob_nms_tensor, output_desc_tensors],
        #                 feed_dict={input_img_tensor: np.expand_dims(img2, 0)})
        # keypoint_map2 = np.squeeze(out2[0])
        # descriptor_map2 = np.squeeze(out2[1])
        # kp2, desc2 = extract_superpoint_keypoints_and_descriptors(
        #         keypoint_map2, descriptor_map2, keep_k_best)

        # # Match and get rid of outliers
        # m_kp1, m_kp2, matches = match_descriptors(kp1, desc1, kp2, desc2)
        # H, inliers = compute_homography(m_kp1, m_kp2)

        # # Draw SuperPoint matches
        # matches = np.array(matches)[inliers.astype(bool)].tolist()
        # matched_img = cv2.drawMatches(img1_orig, kp1, img2_orig, kp2, matches,
        #                               None, matchColor=(0, 255, 0),
        #                               singlePointColor=(0, 0, 255))

        # img1Points = getPointTuplesFromKeyPoints(kp1)
        # print(img1Points[0])
        # print(desc1)
        

        # cv2.imshow("SuperPoint matches", matched_img)
        #
        # # Compare SIFT matches
        # sift_kp1, sift_desc1 = extract_SIFT_keypoints_and_descriptors(img1_orig)
        # sift_kp2, sift_desc2 = extract_SIFT_keypoints_and_descriptors(img2_orig)
        # sift_m_kp1, sift_m_kp2, sift_matches = match_descriptors(
        #         sift_kp1, sift_desc1, sift_kp2, sift_desc2)
        # sift_H, sift_inliers = compute_homography(sift_m_kp1, sift_m_kp2)
        #
        # # Draw SIFT matches
        # sift_matches = np.array(sift_matches)[sift_inliers.astype(bool)].tolist()
        # sift_matched_img = cv2.drawMatches(img1_orig, sift_kp1, img2_orig,
        #                                    sift_kp2, sift_matches, None,
        #                                    matchColor=(0, 255, 0),
        #                                    singlePointColor=(0, 0, 255))
        # cv2.imshow("SIFT matches", sift_matched_img)


        # cv2.waitKey(0)
