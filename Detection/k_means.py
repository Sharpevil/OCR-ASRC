import sys
sys.path.append('C:\\Users\\Zilla\\PyCharmProjects\OCR')
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import Detection.utils as utils


def get_clustered_image(image, clusters):
    # load image, convert to RGB to display with matplotlib
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reshape image to be list of pixels
    image_array = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster pixel intensities
    clt = KMeans(n_clusters=clusters)
    clt.fit(image_array)

    # recolor the original image using the clustered colors
    new_image = utils.recolor_img(image_array, clt)
    new_image = new_image.reshape((image.shape[0], image.shape[1], 3))
    plt.figure()
    plt.axis("off")
    plt.imshow(new_image)
    plt.show()


def get_greyscale_layers(image, clusters):
    # load image, convert to RGB to display with matplotlib
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reshape image to be list of pixels
    image_array = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster pixel intensities
    clt = KMeans(n_clusters=clusters)
    clt.fit(image_array)

    # recolor the original image using the clustered colors
    for cluster in range(0, clusters):
        new_image = utils.layer_img(image_array, clt, cluster)
        new_image = new_image.reshape((image.shape[0], image.shape[1], 3))
        plt.figure()
        plt.axis("off")
        plt.imshow(new_image)
        plt.show()
