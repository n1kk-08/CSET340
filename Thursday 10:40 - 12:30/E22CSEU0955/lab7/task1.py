import cv2
import numpy as np
import matplotlib.pyplot as plt

image_info = [
    ('C:/Users/cchan/OneDrive/Documents/COLLEGE/SEMESTERS/Semester 6/340 l/lab/lab7/a.png', 'Blood Cell Image'),
    ('C:/Users/cchan/OneDrive/Documents/COLLEGE/SEMESTERS/Semester 6/340 l/lab/lab7/b.png', 'Satellite Image')
]

for image_path, title in image_info:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(title, fontsize=18, fontweight='bold')

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    log = cv2.Laplacian(image, cv2.CV_64F)
    axes[0, 1].imshow(log, cmap='gray')
    axes[0, 1].set_title("LoG Blob Detection")
    axes[0, 1].axis('off')

    blur1 = cv2.GaussianBlur(image, (5, 5), 1)
    blur2 = cv2.GaussianBlur(image, (5, 5), 2)
    dog = blur1 - blur2
    axes[0, 2].imshow(dog, cmap='gray')
    axes[0, 2].set_title("DoG Blob Detection")
    axes[0, 2].axis('off')

    alpha = 1.5
    beta = 20
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    axes[1, 0].imshow(enhanced, cmap='gray')
    axes[1, 0].set_title("Brightness & Contrast Adjustment")
    axes[1, 0].axis('off')

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    axes[1, 1].imshow(sharpened, cmap='gray')
    axes[1, 1].set_title("Sharpened Image")
    axes[1, 1].axis('off')

    equalized = cv2.equalizeHist(image)
    axes[1, 2].imshow(equalized, cmap='gray')
    axes[1, 2].set_title("Histogram Equalization")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()