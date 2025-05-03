import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from skimage.feature import hog
from skimage.filters import laplace, difference_of_gaussians


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# -------------------------
# Blob Detection

def hog_blob_detection(image):
    """Histogram of Oriented Gradients (HoG) using skimage's built-in function."""
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, feature_vector=False)

    # Normalize HoG image for better visibility
    hog_image = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX)
    
    return hog_image.astype(np.uint8) 

def log_blob_detection(image):
    """Laplacian of Gaussian (LoG) using skimage's laplace function."""
    log = laplace(cv2.GaussianBlur(image, (5, 5), 0))
    return np.abs(log)  # Convert negative values to positive

def dog_blob_detection(image):
    """Difference of Gaussians (DoG) using skimage's built-in function."""
    return difference_of_gaussians(image, low_sigma=1, high_sigma=2)

def dog_blob_detection(image):
    """Applies Difference of Gaussians (DoG) to detect blobs."""
    return difference_of_gaussians(image, low_sigma=1, high_sigma=2)

def count_candies(image_path):
    """Counts candies using DoG-based blob detection without thresholding."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply DoG only once
    dog_result = dog_blob_detection(image)

    # Normalize DoG image for better visibility
    dog_display = cv2.normalize(dog_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Use SimpleBlobDetector for blob counting
    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByArea = True
    detector_params.minArea = 200  # Adjust to filter small blobs
    detector_params.maxArea = 10000  # Adjust to avoid detecting large noise
    detector_params.filterByCircularity = False  # Disable if candies are irregular
    detector_params.filterByConvexity = False
    detector_params.filterByInertia = False
    
    detector = cv2.SimpleBlobDetector_create(detector_params)
    keypoints = detector.detect(dog_display)

    candy_count = len(keypoints)

    # Draw detected blobs on the original image
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    output = cv2.drawKeypoints(output, keypoints, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1), plt.imshow(dog_display, cmap='gray'), plt.title("DoG Result")
    plt.subplot(1, 2, 2), plt.imshow(output), plt.title("Detected Candies")
    plt.show()

    print(f"Number of candies detected: {candy_count}")
    cv2.imshow("Detected Candies", output)


def process_blob_detection(image_path, is_candy_image=False):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if torch.backends.mps.is_available():
        image = torch.tensor(image, device=device).cpu().numpy()
    log_result = log_blob_detection(image)
    dog_result = dog_blob_detection(image)
    hog_result = hog_blob_detection(image)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1), plt.imshow(log_result, cmap='gray'), plt.title('LoG')
    plt.subplot(1, 3, 2), plt.imshow(dog_result, cmap='gray'), plt.title('DoG')
    plt.subplot(1, 3, 3), plt.imshow(hog_result, cmap='gray'), plt.title('HoG')
    plt.show()
    
    # if is_candy_image:
    #     candy_count = count_candies(image_path)
    #     print(f"Number of candies detected: {candy_count}")

# Calling Blob Detection Functions on Images
process_blob_detection("Blob_1.jpg")
process_blob_detection("Blob_2.png")
# process_blob_detection("Blob_3.png")
process_blob_detection("Candy.jpg", is_candy_image=True)
count_candies("Candy.jpg")

# -------------------------
# Image Quality Enhancement
# -------------------------
def enhance_image(image_path):
    image = cv2.imread(image_path)

    # Adjust brightness & contrast
    enhanced = cv2.convertScaleAbs(image, alpha=1.2, beta=30)

    # Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Noise Removal
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)

    # Histogram Equalization (Color)
    img_yuv = cv2.cvtColor(denoised, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # **Inverse Transform (Negative Image)**
    inverse = cv2.bitwise_not(equalized)

    # # **Super-resolution using OpenCV DNN**
    # sr = cv2.dnn_superres.DnnSuperResImpl_create()
    # sr.readModel("EDSR_x4.pb")  # Requires a pre-trained model
    # sr.setModel("edsr", 4)  # EDSR model with 4x upscaling
    # super_res = sr.upsample(equalized)

    # **Color Correction (Gray World Assumption)**
    avg_b = np.mean(equalized[:, :, 0])
    avg_g = np.mean(equalized[:, :, 1])
    avg_r = np.mean(equalized[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3
    scale_b, scale_g, scale_r = avg_gray / avg_b, avg_gray / avg_g, avg_gray / avg_r
    color_corrected = np.copy(equalized)
    color_corrected[:, :, 0] = np.clip(equalized[:, :, 0] * scale_b, 0, 255)
    color_corrected[:, :, 1] = np.clip(equalized[:, :, 1] * scale_g, 0, 255)
    color_corrected[:, :, 2] = np.clip(equalized[:, :, 2] * scale_r, 0, 255)
    
    # Display Results
    plt.figure(figsize=(15, 5))
    plt.subplot(2, 3, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(2, 3, 2), plt.imshow(cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB)), plt.title('Equalized')
    plt.subplot(2, 3, 3), plt.imshow(cv2.cvtColor(inverse, cv2.COLOR_BGR2RGB)), plt.title('Inverse Transform')
    plt.subplot(2, 3, 5), plt.imshow(cv2.cvtColor(color_corrected, cv2.COLOR_BGR2RGB)), plt.title('Color Corrected')
    plt.show()
    
enhance_image("Enhance.jpg")