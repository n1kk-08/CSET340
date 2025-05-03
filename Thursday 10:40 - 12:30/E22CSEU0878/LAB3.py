import cv2
import numpy as np
from skimage.filters import sobel, prewitt, roberts
import matplotlib.pyplot as plt

def show_image(title, img, wait_time=3000):
    cv2.imshow(title, img)
    cv2.waitKey(wait_time)
    cv2.destroyAllWindows()

def task_1(image_path):
    image = cv2.imread(image_path)
    
    # cv2.imshow('Original Image', image)
    # cv2.waitKey(3000)
    
    height, width, channels = image.shape
    print(f"Image Size: {width}x{height}, Channels: {channels}")
    
    total_pixels = height * width
    print(f"Total Pixels: {total_pixels}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('image_rgb.jpg', image_rgb)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('image_gray.jpg', gray)
    
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite('image_binary.jpg', binary)
    
    black_pixels = np.sum(binary == 0)
    print(f"Black pixel area: {black_pixels}")
    
    cv2.destroyAllWindows()

def task_2(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    sobel_edges = sobel(image)
    prewitt_edges = prewitt(image)
    roberts_edges = roberts(image)
    canny_edges = cv2.Canny(image, 100, 200)
    
    cv2.imwrite('sobel.jpg', (sobel_edges * 255).astype(np.uint8))
    cv2.imwrite('prewitt.jpg', (prewitt_edges * 255).astype(np.uint8))
    cv2.imwrite('roberts.jpg', (roberts_edges * 255).astype(np.uint8))
    cv2.imwrite('canny.jpg', canny_edges)
    
    _, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    markers = cv2.connectedComponents(sure_fg)[1]
    markers += 1
    markers[unknown == 255] = 0
    watershed_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.watershed(watershed_image, markers)
    watershed_image[markers == -1] = [0, 0, 255]
    
    cv2.imwrite('global_threshold.jpg', global_thresh)
    cv2.imwrite('adaptive_threshold.jpg', adaptive_thresh)
    cv2.imwrite('watershed.jpg', watershed_image)

task_1('GrayScale.png')
task_2('GrayScale.png')
