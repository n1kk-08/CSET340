import cv2
import matplotlib.pyplot as plt

imgs = [cv2.imread('C:/Users/cchan/OneDrive/Documents/COLLEGE/SEMESTERS/Semester 6/340 l/lab/lab9/x.jpg'), 
        cv2.imread('C:/Users/cchan/OneDrive/Documents/COLLEGE/SEMESTERS/Semester 6/340 l/lab/lab9/y.jpg'),
        cv2.imread('C:/Users/cchan/OneDrive/Documents/COLLEGE/SEMESTERS/Semester 6/340 l/lab/lab9/z.jpg')]

for i, img in enumerate(imgs):
    if img is None:
        print(f"Failed to load image {i+1}.jpg")
        exit()
    print(f"Image {i+1} shape: {img.shape}")

imgs_resized = [cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)) for img in imgs]

sift = cv2.SIFT_create()
keypoints_list = []
descriptors_list = []

for img in imgs_resized:
    keypoints, descriptors = sift.detectAndCompute(img, None)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)


bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches_all = []

for i in range(len(imgs_resized) - 1):
    matches = bf.match(descriptors_list[i], descriptors_list[i + 1])
    matches = sorted(matches, key=lambda x: x.distance)
    matches_all.append(matches)

    img_matches = cv2.drawMatches(imgs_resized[i], keypoints_list[i], imgs_resized[i + 1], keypoints_list[i + 1], matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    plt.figure(figsize=(10, 5))
    plt.imshow(img_matches)
    plt.title(f"Keypoint Matches Between Image {i+1} and Image {i+2}")
    plt.axis('off')
    plt.show()


stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS) 
status, pano = stitcher.stitch(imgs_resized)


fig = plt.figure(figsize=(20, 10))

for i, img in enumerate(imgs_resized):
    ax = fig.add_subplot(2, 3, i+1)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Image {i+1}")
    ax.axis('off')

if status == cv2.Stitcher_OK:
    ax = fig.add_subplot(2, 1, 2)
    ax.imshow(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB))
    ax.set_title("Stitched Panorama")
    ax.axis('off')
    cv2.imwrite("panorama.jpg", pano)
else:
    print(f"Stitching failed. Status code: {status}")

plt.tight_layout()
plt.show()