import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

path1 = 'C:/Users/cchan/OneDrive/Documents/COLLEGE/SEMESTERS/Semester 6/340 l/lab/lab9/b1.png'
path2 = 'C:/Users/cchan/OneDrive/Documents/COLLEGE/SEMESTERS/Semester 6/340 l/lab/lab9/b2.png'

if not os.path.exists(path1) or not os.path.exists(path2):
    print("Error: One or both image files not found. Check file paths.")
    exit()

img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: One or both images could not be loaded. Check file format and integrity.")
    exit()

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)[:100]

img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

plt.figure(figsize=(15, 8))
plt.title("Feature Matches")
plt.imshow(img_matches, cmap='gray')
plt.axis("off")
plt.show()

pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

F, mask_f = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
print("Fundamental Matrix:\n", F)

K = np.array([[718.8560, 0, 607.1928],
              [0, 718.8560, 185.2157],
              [0, 0, 1]])

E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
print("Essential Matrix:\n", E)

_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
print("Rotation Matrix (R):\n", R)
print("Translation Vector (t):\n", t)