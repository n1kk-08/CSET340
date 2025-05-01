    # <Image segmentation>

import cv2
import numpy as np

def main():
    image_path = 'l2.jpeg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Original Image in GRAYSCALE', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    if image is None:
        print("Error: Image not found!")
        return
    
    # Global Thresholding
    _, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('Global Threshold', global_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imshow('Adaptive Threshold', adaptive_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Edge Detection for Segmentation (Canny)
    edges = cv2.Canny(image, 100, 200)
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Watershed Algorithm (Region-Based Segmentation)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    image_color = cv2.imread(image_path)
    markers = cv2.watershed(image_color, markers)
    image_color[markers == -1] = [255, 0, 0]

    cv2.imshow('Watershed Segmentation', image_color) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# USE EROSION FOR WATERSHED 