import cv2
import numpy as np
import matplotlib.pyplot as plt

def blur_image(image, method, ksize=5):
    if method == 'box':
        return cv2.blur(image, (ksize, ksize))
    elif method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'adaptive':
        return cv2.bilateralFilter(image, 9, 75, 75)

def display_images(original, transformed, title):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    image = cv2.imread('lenna.png')

    blur_box = blur_image(image, 'box', 7)
    display_images(image, blur_box, 'Box Blurring')

    blur_gaussian = blur_image(image, 'gaussian', 7)
    display_images(image, blur_gaussian, 'Gaussian Blurring')

    blur_adaptive = blur_image(image, 'adaptive')
    display_images(image, blur_adaptive, 'Adaptive Blurring')
