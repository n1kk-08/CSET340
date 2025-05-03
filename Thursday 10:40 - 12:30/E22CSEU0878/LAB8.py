import cv2
import numpy as np
import matplotlib.pyplot as plt

def sift_interest_points(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    output_image = cv2.drawKeypoints(image, keypoints, None)
    
    plt.imshow(output_image, cmap='gray')
    plt.title("SIFT Interest Points")
    plt.show()

def orb_feature_matching(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.imshow(result)
    plt.title("Feature Matching with ORB and BFMatcher")
    plt.show()

def contour_detection(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Unable to read image. Check path and format.")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on a black background
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)

    # Display results
    plt.figure(figsize=(8,6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(contour_image, cmap='gray')
    plt.title("Contour Detection")

    plt.show()

# Call statements for Task 1
sift_interest_points("sift_img.png")
orb_feature_matching("img1_1.png", "img1_2.png")
contour_detection("contour_img.png")
