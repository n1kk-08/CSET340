import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'C:/Users/cchan/OneDrive/Documents/COLLEGE/SEMESTERS/Semester 6/340 l/lab/lab7/c.png'

img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def adjust_brightness_contrast(image, alpha=1.5, beta=30):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def enhance_color(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def resize_image(image, scale_percent=150):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

def inverse_transform(image):
    return cv2.bitwise_not(image)

def equalize_histogram(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def super_resolution(image):
    return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)

def color_correction(image):
    result = image.copy()
    avg_b = np.average(result[:, :, 0])
    avg_g = np.average(result[:, :, 1])
    avg_r = np.average(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3
    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / avg_b), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / avg_g), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / avg_r), 0, 255)
    return result.astype(np.uint8)

bright_contrast = adjust_brightness_contrast(img)
sharpened = sharpen_image(img)
denoised = denoise_image(img)
color_enhanced = enhance_color(img)
resized = resize_image(img)
inversed = inverse_transform(img)
equalized = equalize_histogram(img)
super_res = super_resolution(img)
color_corrected = color_correction(img)

titles = [
    'Original', 'Brightness & Contrast', 'Sharpened', 'Denoised',
    'Color Enhanced', 'Resized', 'Inverse Transform', 'Equalized Histogram',
    'Super Resolution', 'Color Corrected'
]

images = [
    img_rgb,
    cv2.cvtColor(bright_contrast, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(color_enhanced, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(inversed, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(super_res, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(color_corrected, cv2.COLOR_BGR2RGB)
]

plt.figure(figsize=(18, 14), facecolor='white')
for i in range(len(images)):
    plt.subplot(4, 3, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i], fontsize=12)
    plt.axis('off')

for j in range(len(images), 12):
    plt.subplot(4, 3, j + 1)
    plt.axis('off')

plt.tight_layout()
plt.show()