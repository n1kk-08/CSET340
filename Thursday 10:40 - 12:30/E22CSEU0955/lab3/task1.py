import cv2
import numpy as np

def resize_image(image, scale_percent=30):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height))

def main():
    #-1 Read the image
    image_path = 'l3.jpg' 
    image = cv2.imread(image_path)
     
    if image is None:
        print("Error: Image not found!")
        return
    
    resized_image = resize_image(image)
    
    # <Original image>
    cv2.imshow('Original Image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #-2 BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_rgb = resize_image(image_rgb)
    
    # <RGB image>
    cv2.imshow('RGB Image', resized_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #-3 Extraction of Image
    height, width, channels = image.shape
    print(f'Image Size: {width}x{height}, Channels: {channels}')
    
    #-4 Calculate total image pixels
    total_pixels = width * height
    print(f'Total Pixels: {total_pixels}')
    
    #-5 RGB to Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_gray = resize_image(gray_image)
    
    # <Grayscale image>
    cv2.imshow('Grayscale Image', resized_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #-6 Convert Grayscale to Binary Image using user-defined threshold
    threshold_value = int(input("Enter threshold value (0-255): "))
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    resized_binary = resize_image(binary_image)
    
    # <Binary image>
    cv2.imshow('Binary Image', resized_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #-7 Count black pixels in binary image
    black_pixels = np.sum(binary_image == 0)
    print(f'Black Pixels Count: {black_pixels}')
    
if __name__ == "__main__":
    main()