import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_image(image, method, size=(200, 200)):
    if method == 'linear':
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    elif method == 'nearest':
        return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    elif method == 'cubic':
        return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

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

    resized_linear = resize_image(image, 'linear', (200, 200))
    display_images(image, resized_linear, 'Resized (Linear)')

    resized_nearest = resize_image(image, 'nearest', (200, 200))
    display_images(image, resized_nearest, 'Resized (Nearest)')

    resized_cubic = resize_image(image, 'cubic', (200, 200))
    display_images(image, resized_cubic, 'Resized (Cubic)')
