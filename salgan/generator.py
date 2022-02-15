from operator import gt
from unittest.mock import patch
from attr import has
import cv2
import numpy as np

s_smooth = 1
g_ths = [20, 20, 20]
has_selected = False
keypoints = []
num_sel = 0

def generateKeypoints(image_gs, heatmap, num_points=1000):
    global s_smooth, g_ths, has_selected, keypoints, num_sel, total_weight
    keypoints = []
    grad_image = computeGradient(image_gs)
    grad_heatmap = computeGradient(heatmap)
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
            keypoints.append(c[1])
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

def computeGradient(heatmap):
    grad = np.gradient(heatmap)
    x_grad = grad[0]
    y_grad = grad[1]
    total_grad = np.sqrt((x_grad*x_grad) + (y_grad*y_grad))
    return total_grad


def quarterPatch(p):
    hs = (int)(p.shape[0]/2)
    return [p[:hs,:hs],p[:hs,hs:],p[hs:,:hs],p[hs:,hs:]]





def generateDescriptor(keypoints):
    pass