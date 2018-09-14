# imports of import
import numpy as np
import cv2
from skimage import feature


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=num_labels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def color_bar(hist, centroids):
    # initialize the bar chart representing the relative frequency of each color
    bar = np.zeros((50, 300, 3), dtype="uint8")
    start_x = 0

    # loop over the percentage of each cluster and the color of each cluster
    for(percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        end_x = start_x + (percent * 300)
        cv2.rectangle(bar, (int(start_x), 0), (int(end_x), 50), color.astype("uint8").tolist(), -1)
        start_x = end_x

    return bar


def recolor_img(image, clt):
    for center in clt.cluster_centers_:
        print(center)
    for point in range(0, len(image) - 1):
        image[point] = clt.cluster_centers_[clt.labels_[point]]
    return image


def layer_img(image, clt, layer):
    for point in range(0, len(image) - 1):
        if clt.labels_[point] == layer:
            image[point] = clt.cluster_centers_[clt.labels_[point]]
        else:
            image[point] = [256, 256, 256]
    return image


# Automatic threshold generation taken from
# https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def canny_edge(image, sigma=0.33):
    image = cv2.imread(image, 0)
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(image, lower, upper)
    return edges
