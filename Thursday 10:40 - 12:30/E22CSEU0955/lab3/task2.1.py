    # <Edge detection>

import cv2
import numpy as np
from scipy import ndimage

def main():
    image_path = 'l2.jpeg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Original Image in grayscale', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if image is None:
        print("Error: Image not found!")
        return
    
    #-a Sobel Operator
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edge = cv2.magnitude(sobel_x, sobel_y)
    cv2.imshow('Sobel Edge', sobel_edge.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    #-b Prewitt Operator
    prewitt_x = ndimage.convolve(image, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
    prewitt_y = ndimage.convolve(image, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
    prewitt_edge = np.hypot(prewitt_x, prewitt_y)
    cv2.imshow('Prewitt Edge', prewitt_edge.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #-c Roberts Cross Operator
    roberts_x = ndimage.convolve(image, np.array([[1, 0], [0, -1]]))
    roberts_y = ndimage.convolve(image, np.array([[0, 1], [-1, 0]]))
    roberts_edge = np.hypot(roberts_x, roberts_y)
    cv2.imshow('Roberts Edge', roberts_edge.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #-d Canny Edge Detector
    canny_edge = cv2.Canny(image, 100, 200)
    cv2.imshow('Canny Edge', canny_edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()