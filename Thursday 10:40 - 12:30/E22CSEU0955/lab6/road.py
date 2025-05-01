import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from ultralytics import YOLO
import torch
import torchvision
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")

image_path = 'C:/Users/cchan/OneDrive/Documents/COLLEGE/SEMESTERS/Semester 6/340 l/lab/lab6/road.png'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h_channel = hsv[:, :, 0]

kernel = np.ones((5, 5), np.uint8)
morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

def apply_gabor(img, ksize=31, sigma=4, theta=np.pi/4, lambd=10, gamma=0.5):
    g_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
    return filtered_img

gabor_filtered = apply_gabor(morph)

radius = 1
n_points = 8 * radius
lbp = local_binary_pattern(morph, n_points, radius, method="uniform")

edges = cv2.Canny(morph, 50, 150)

lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=5)
line_image = np.copy(image)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=100)
circle_image = np.copy(image)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(circle_image, (i[0], i[1]), i[2], (0, 255, 0), 2)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

titles = [
    "Original Image", "Hue Channel (HSV)", "Morphological Operations", "Gabor Filtered",
    "Local Binary Pattern (LBP)", "Canny Edges", "Hough Lines", "Hough Circles"
]
images = [
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB), h_channel, morph, gabor_filtered,
    lbp, edges, cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB), cv2.cvtColor(circle_image, cv2.COLOR_BGR2RGB)
]
cmaps = ['RGB', 'gray', 'gray', 'gray', 'gray', 'gray', 'RGB', 'RGB']

for idx, ax in enumerate(axes):
    if cmaps[idx] == 'RGB':
        ax.imshow(images[idx])
    else:
        ax.imshow(images[idx], cmap='gray')
    ax.set_title(titles[idx])
    ax.axis('off')

plt.suptitle('Feature Extraction and Detection Results', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# YOLOv8 Detection 

yolo = YOLO('yolov8n.pt')

results = yolo(image)

annotated_image = results[0].plot()
plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.title('YOLOv8 Object Detection')
plt.axis('off')
plt.show()

# Faster R-CNN Detection

transform = transforms.Compose([
    transforms.ToTensor()
])

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

image_tensor = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0)

with torch.no_grad():
    predictions = model(image_tensor)
    
rcnn_image = np.copy(image)
for i, box in enumerate(predictions[0]['boxes']):
    score = predictions[0]['scores'][i].item()
    if score > 0.5:
        x1, y1, x2, y2 = map(int, box.numpy())
        cv2.rectangle(rcnn_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(rcnn_image, cv2.COLOR_BGR2RGB))
plt.title('Faster R-CNN Object Detection')
plt.axis('off')
plt.show()