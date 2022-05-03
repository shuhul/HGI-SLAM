from curses import raw
import handler
import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.stats as st


if __name__ == "__main__":
    # folder = '/root/KITTI_06' #00 02 05 06 07 09
    # folder = '/root/MONO_DESK'
    # handler.readFolder(folder)
    # lcc = [(26,857), (171, 997)]
    # lcc = [(14,51)]
    # handler.showMap(lcc, isfr=True)
    # handler.showLoopClosurePairs(lcc)

    # handler.showVideo(skip=10)

    simsFile = open('sims', 'rb')
    rawsims = pickle.load(simsFile)

    sims = []
    for s in rawsims:
        if s < 1:
            sims.append(s)




    # plt.hist(sims, bins=50, cmap=)

    scale = 1/23
    n, bins, patches = plt.hist(sims, bins=15, weights=scale*np.ones_like(sims))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)

    cm = plt.cm.get_cmap('Oranges')

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))

    sp = handler.col(173, 231, 5, 1.0)

    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(sims)
    scale2 = 0.33
    plt.plot(kde_xs, scale2*kde.pdf(kde_xs), color=sp)

    plt.ylabel('Relative Magnitude')
    plt.xlabel('Similarity')
    plt.savefig('sim.png')

    img = cv2.imread('sim.png', cv2.IMREAD_COLOR)
    img = cv2.bitwise_not(img)
    cv2.imwrite('sim.png', img)
