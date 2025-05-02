import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(title, image, is_gray=False):
    plt.figure(figsize=(6, 6))
    if is_gray:
        plt.imshow(image, cmap='gray')
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

img_saliency = cv2.imread('C:/Users/cchan/OneDrive/Documents/COLLEGE/SEMESTERS/Semester 6/340 l/lab/lab11/sample.png')
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
(success, saliencyMap) = saliency.computeSaliency(img_saliency)
saliencyMap = (saliencyMap * 255).astype("uint8")

show_image("Original - Saliency", img_saliency)
show_image("Saliency Map", saliencyMap, is_gray=True)

image_kmeans = cv2.imread('C:/Users/cchan/OneDrive/Documents/COLLEGE/SEMESTERS/Semester 6/340 l/lab/lab11/landscape.jpg')
Z = image_kmeans.reshape((-1, 3)).astype(np.float32)

K = 4
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
segmented = centers[labels.flatten()].reshape(image_kmeans.shape).astype('uint8')

show_image("Original - K-Means", image_kmeans)
show_image("K-means Segmentation", segmented)

img_grabcut = cv2.imread('C:/Users/cchan/OneDrive/Documents/COLLEGE/SEMESTERS/Semester 6/340 l/lab/lab11/object.png')
mask = np.zeros(img_grabcut.shape[:2], np.uint8)
bgModel = np.zeros((1, 65), np.float64)
fgModel = np.zeros((1, 65), np.float64)
rect = (50, 50, 250, 250)

cv2.grabCut(img_grabcut, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
result = img_grabcut * mask2[:, :, np.newaxis]

show_image("Original - GrabCut", img_grabcut)
show_image("Graph Cut Result", result)