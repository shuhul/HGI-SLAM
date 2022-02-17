from operator import gt
from unittest.mock import patch
from attr import has
import cv2
import numpy as np
from cv2 import KeyPoint
from numpy.linalg import norm
from scipy.interpolate import interp1d
import pickle
# from numba import jit, cuda
# import code

# code.interact(local=locals)

s_smooth = 1
g_ths = [20, 20, 20]
has_selected = False
keypoints = []
num_sel = 0


def generateKeypoints(image_gs, heatmap, num_points=1000):
    global s_smooth, g_ths, has_selected, keypoints, num_sel, total_weight
    keypoints = []
    grad_image, oren_image = computeGradients(image_gs)
    grad_heatmap, oren_heatmap = computeGradients(heatmap)
    patches, weights = genPatches(grad_image, grad_heatmap)
    num_sel = 0
    while num_sel < num_points:
        m = getPatch(patches,weights)
        has_selected = False
        for d4 in quarterPatch(m):
            for d2 in quarterPatch(d4):
                for d in quarterPatch(d2):
                    selectKeypoint(d, 0)
                selectKeypoint(d2, 1)
            selectKeypoint(d4, 2)
    return keypoints


def selectKeypoint(patch, thresh):
    global g_ths, keypoints, has_selected, num_sel
    if not has_selected:
        c = max([patch.flatten()], key=lambda item:item[0])
        if c[0] > g_ths[thresh]:
            keypoints.append(KeyPoint(c[1][0], c[1][1], c[0]))
            # keypoints.append(c[1])
            num_sel += 1
            has_selected = True
            

def getPatch(patches, weights):
    indices = list(range(0,len(patches)))
    return patches[np.random.choice(indices, p=weights)]

def genSampleWeight(patch):
    return (np.median(patch.flatten()) + s_smooth)

def genPatches(g_i, g_h, K=8):
    patches = []
    weights = []
    total_weight = 0
    for x in range(0, g_i.shape[0], K):
        for y in range(0, g_i.shape[1], K):
            patch_h = g_h[x:(x+K), y:(y+K)]
            weight = genSampleWeight(patch_h)
            weights.append(weight)
            total_weight += weight

            patch_i = g_i[x:(x+K), y:(y+K)]
            patch = np.array([[[0,(0,0)]]*K]*K, dtype=object)
            for i in range(K):
                for j in range(K):
                    patch[i][j] = [patch_i[i][j], (y+j, x+i)]
            patches.append(patch)
    return patches, (weights/total_weight)

def computeGradients(map):
    grad = np.gradient(map)
    x_grad = grad[0]
    y_grad = grad[1]
    total_grad = np.sqrt((x_grad*x_grad) + (y_grad*y_grad))
    oren_grad = np.rad2deg(np.arctan2(y_grad, x_grad)) % 360
    return total_grad, oren_grad


def quarterPatch(p):
    hs = (int)(p.shape[0]/2)
    return [p[:hs,:hs],p[:hs,hs:],p[hs:,:hs],p[hs:,hs:]]

def kpsToKPS(kps):
    KPS = []
    for kp in kps:
        KPS.append(KeyPoint(kp[0], kp[1], 1))
    return KPS

def smoothHist(hist, weight=0.3):
    smooth_hist = [0.0]*len(hist)
    f = interp1d(range(len(hist)), hist, kind='cubic')
    smooth_hist[0] = (f(0) + f(weight))/2
    smooth_hist[len(hist)-1] = (f(len(hist)-1-weight) + f(len(hist)-1))/2
    for i in range(1, len(hist)-1):
        smooth_hist[i] = (f(i-weight) + f(i+weight))/2
    return smooth_hist

def generateDescriptors(image_gs, keypoints):
    image_blur = cv2.GaussianBlur(image_gs, (5, 5), 2)
    image_w = image_blur.shape[0]
    image_h = image_blur.shape[1]
    descriptors = []
    for keypoint in keypoints:
        point = keypoint.pt
        px = int(point[0])
        py = int(point[1])
        descriptor_vector = []
        if px > 8 and py > 8 and px < image_w-8 and py < image_h-8:
            patch = image_blur[(px-8):(px+8), (py-8):(py+8)]
            kp_mag, kp_oren = computeGradients(patch)
            kp_mag = kp_mag[7,7]
            kp_oren = kp_oren[7,7]
            all_hist = []
            for qp in quarterPatch(patch):
                for block in quarterPatch(qp):
                    oren_hist = np.array([0.0]*8)
                    mag, oren = computeGradients((block-kp_oren)%360)
                    mag = mag.flatten()
                    oren = oren.flatten()
                    for i in range(len(oren)):
                        oren_hist[int(oren[i]//45)] += mag[i]
                    # oren_hist = smoothHist(oren_hist)
                    all_hist.append(oren_hist)
            all_hist = np.array(all_hist, dtype = 'float32')

            all_hist = all_hist.reshape((8,4,4))
            for i in range(len(all_hist)):
                all_hist[i] = cv2.GaussianBlur(all_hist[i], (5, 5), 0.5, 0.5)
                
            descriptor_vector = all_hist.flatten()
            threshold = norm(descriptor_vector) * 0.2
            descriptor_vector[descriptor_vector > threshold] = threshold
            descriptor_vector /= max(norm(descriptor_vector), 1e-7)
            descriptor_vector = np.round(512 * descriptor_vector)
            descriptor_vector[descriptor_vector <= 0] = 0
            descriptor_vector[descriptor_vector > 255] = 255
            descriptors.append(descriptor_vector)

    return np.array(descriptors, dtype = 'float32')

def generateKeypointsAndDescriptors(image_gs, heatmap):
    keypoints = generateKeypoints(image_gs, heatmap)
    # x = add(1,2)
    # pickle.dump(keypoints, open("keypoints", "wb"))
    # keypoints = kpsToKPS(pickle.load(open("keypoints", "rb")))
    descriptors = generateDescriptors(image_gs, keypoints)
    return keypoints, descriptors

# @jit(nopython=True)
# def add(a,b):
#     return a+b